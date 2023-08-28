// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include "kernels/funcs/format_utils.h"
#include "acltransformer/params/norm.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "acltransformer/plan.h"
#include "acltransformer/statistic.h"
#include "acltransformer/ops/softmax_operation.h"
#include "kernels/funcs/format_utils.h"
#endif

struct SoftmaxWorkspace {
  void *workspace_ = nullptr;
  uint64_t workspaceSize_ = 0;
};

SoftmaxWorkspace g_softmaxWorkSpace = {nullptr, 0};
std::unique_ptr<AclTransformer::SoftmaxOperation> g_softmaxOp;
std::unique_ptr<AclTransformer::Plan> g_softmaxPlan;

std::vector<std::vector<int64_t>> SoftmaxOpInferShape(
    const std::vector<int64_t> &input_x_shape,
    int axes) {

  return {input_x_shape};
}

static void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                             std::vector<const phi::DenseTensor *> &outTensors,
                             AclTransformer::VariantPack &variantPack) {
  variantPack.inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); ++i) {
    variantPack.inTensors.at(i) =
        ConvertDenseTensorToAsdTensor(*(inTensors.at(i)));
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
        variantPack.inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
      variantPack.inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  }

  variantPack.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); ++i) {
    variantPack.outTensors.at(i) =
        ConvertDenseTensorToAsdTensor(*(outTensors.at(i)));
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
        variantPack.outTensors.at(i).desc.format ==
            AsdOps::TENSOR_FORMAT_NCHW) {
      variantPack.outTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  }
}

static void SetWorkspace(uint64_t workspaceSize) {
  if (workspaceSize <= g_softmaxWorkSpace.workspaceSize_) {
    VLOG(6) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_softmaxWorkSpace.workspaceSize_
            << ", not new device mem";
    return;
  }

  if (g_softmaxWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_softmaxWorkSpace.workspace_);
    g_softmaxWorkSpace.workspace_ = nullptr;
    g_softmaxWorkSpace.workspaceSize_ = 0;
  }

  VLOG(6) << "softmaxOp SetWorkspace AsdRtMemMallocDevice workspaceSize:"
          << workspaceSize;
  int st = AsdRtMemMallocDevice((void **)&(g_softmaxWorkSpace.workspace_),
                                workspaceSize,
                                ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("SoftmaxOp SetWorkspace AsdRtMemMallocDevice,"
                            "fail, ret: %d .",
                            st));

  g_softmaxWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_softmaxWorkSpace.workspace_; }

std::vector<paddle::Tensor> SoftmaxOp(const paddle::Tensor &input_x,
                                   int axes) {
  std::cout << "run in SoftmaxOp" << std::endl;
  std::cout << "num = " << axes << std::endl;

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  auto input_x_tensor =
      static_cast<const phi::DenseTensor *>(input_x.impl().get());

  auto out_shape = SoftmaxOpInferShape(input_x.shape(), num).at(0);

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), input_x_tensor->dtype());
  if (!g_softmaxOp) {
    AclTransformer::SoftmaxParam param = {axes};
    g_softmaxOp.reset(new AclTransformer::SoftmaxOperation(param));
    g_softmaxPlan.reset(new AclTransformer::Plan);
    g_softmaxOp->BuildPlan(g_softmaxPlan.get());
  }
  AclTransformer::VariantPack variantPack;
  std::vector<const phi::DenseTensor *> inputs = {input_x_tensor};
  std::vector<const phi::DenseTensor *> outputs = {out_tensor.get()};
  BuildVariantPack(inputs, outputs, variantPack);

  AsdOps::Status st = g_softmaxPlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("softmaxPlan Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));
  variantPack.workspaceSize = g_softmaxPlan->GetWorkspaceSize();
  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }
  st = g_softmaxPlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(
      st.Ok(),
      true,
      phi::errors::External("g_softmaxPlan Execute plan failed,"
                            "ret message: %s .",
                            st.Message()));
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(softmax_op)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"axes: int"})
    .SetKernelFn(PD_KERNEL(SoftmaxOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        SoftmaxOpInferShape));  // neccessary if the op has muti_inputs