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
#include "acltransformer/ops/multinomial_operation.h"
#include "kernels/funcs/format_utils.h"
#endif

struct MultinomialWorkspace {
  void *workspace_ = nullptr;
  uint64_t workspaceSize_ = 0;
};

MultinomialWorkspace g_multinomialWorkSpace = {nullptr, 0};
std::unique_ptr<AclTransformer::MultinomialOperation> g_multinomialOp;
std::unique_ptr<AclTransformer::Plan> g_multinomialPlan;

std::vector<std::vector<int64_t>> MultinomialOpInferShape(
    const std::vector<int64_t> &input_x_shape,
    int num_samples) {
  std::vector<int64_t> x_shape = input_x_shape;

  std::vector<int64_t> out_dims;
  out_dims[0] = x_shape[0]
  out_dims.push_back(num_samples);

  return {out_dims};
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
  if (workspaceSize <= g_multinomialWorkSpace.workspaceSize_) {
    VLOG(6) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_multinomialWorkSpace.workspaceSize_
            << ", not new device mem";
    return;
  }

  if (g_multinomialWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_multinomialWorkSpace.workspace_);
    g_multinomialWorkSpace.workspace_ = nullptr;
    g_multinomialWorkSpace.workspaceSize_ = 0;
  }

  VLOG(6) << "multinomialOp SetWorkspace AsdRtMemMallocDevice workspaceSize:"
          << workspaceSize;
  int st = AsdRtMemMallocDevice((void **)&(g_multinomialWorkSpace.workspace_),
                                workspaceSize,
                                ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("MultinomialOp SetWorkspace AsdRtMemMallocDevice,"
                            "fail, ret: %d .",
                            st));

  g_multinomialWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_multinomialWorkSpace.workspace_; }

std::vector<paddle::Tensor> MultinomialOp(const paddle::Tensor &input_x,
                                   int num_samples) {
  std::cout << "run in MultinomialOp" << std::endl;
  std::cout << "num_samples = " << num << std::endl;

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  auto input_x_tensor =
      static_cast<const phi::DenseTensor *>(input_x.impl().get());

  auto out_shape = MultinomialOpInferShape(input_x.shape(), num).at(0);

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), input_x_tensor->dtype());
  if (!g_multinomialOp) {
    AclTransformer::MultinomialParam param = {num_samples};
    g_multinomialOp.reset(new AclTransformer::MultinomialOperation(param));
    g_multinomialPlan.reset(new AclTransformer::Plan);
    g_multinomialOp->BuildPlan(g_multinomialPlan.get());
  }
  AclTransformer::VariantPack variantPack;
  std::vector<const phi::DenseTensor *> inputs = {input_x_tensor};
  std::vector<const phi::DenseTensor *> outputs = {out_tensor.get()};
  BuildVariantPack(inputs, outputs, variantPack);

  AsdOps::Status st = g_multinomialPlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("multinomialPlan Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));
  variantPack.workspaceSize = g_multinomialPlan->GetWorkspaceSize();
  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }
  st = g_multinomialPlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(
      st.Ok(),
      true,
      phi::errors::External("g_multinomialPlan Execute plan failed,"
                            "ret message: %s .",
                            st.Message()));
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(multinomial_op)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"num_samples: uint32_t"})
    .SetKernelFn(PD_KERNEL(MultinomialOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MultinomialOpInferShape));  // neccessary if the op has muti_inputs