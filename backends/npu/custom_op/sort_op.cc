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
#include "acltransformer/ops/sort_operation.h"
#include "kernels/funcs/format_utils.h"
#endif

struct SortWorkspace {
  void *workspace_ = nullptr;
  uint64_t workspaceSize_ = 0;
};

SortWorkspace g_sortWorkSpace = {nullptr, 0};
std::unique_ptr<AclTransformer::SortOperation> g_sortOp;
std::unique_ptr<AclTransformer::Plan> g_sortPlan;

std::vector<std::vector<int64_t>> SortOpInferShape(
    const std::vector<int64_t> &input_x_shape,
    int num) {
  std::vector<int64_t> x_shape = input_x_shape;

  std::vector<int64_t> out_dims;
  out_dims.assign(x_shape.begin(), x_shape.end());
  out_dims.push_back(num);

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
  if (workspaceSize <= g_sortWorkSpace.workspaceSize_) {
    VLOG(6) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_sortWorkSpace.workspaceSize_
            << ", not new device mem";
    return;
  }

  if (g_sortWorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_sortWorkSpace.workspace_);
    g_sortWorkSpace.workspace_ = nullptr;
    g_sortWorkSpace.workspaceSize_ = 0;
  }

  VLOG(6) << "sortOp SetWorkspace AsdRtMemMallocDevice workspaceSize:"
          << workspaceSize;
  int st = AsdRtMemMallocDevice((void **)&(g_sortWorkSpace.workspace_),
                                workspaceSize,
                                ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(
      st,
      ASDRT_SUCCESS,
      phi::errors::External("SortOp SetWorkspace AsdRtMemMallocDevice,"
                            "fail, ret: %d .",
                            st));

  g_sortWorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_sortWorkSpace.workspace_; }

std::vector<paddle::Tensor> SortOp(const paddle::Tensor &input_x,
                                   int num) {
  std::cout << "run in SortOp" << std::endl;
  std::cout << "num = " << num << std::endl;

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  auto input_x_tensor =
      static_cast<const phi::DenseTensor *>(input_x.impl().get());

  auto out_shape = SortOpInferShape(input_x.shape(), num).at(0);

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), input_x_tensor->dtype());
  if (!g_sortOp) {
    AclTransformer::SortParam param = {num};
    g_sortOp.reset(new AclTransformer::SortOperation(param));
    g_sortPlan.reset(new AclTransformer::Plan);
    g_sortOp->BuildPlan(g_sortPlan.get());
  }
  AclTransformer::VariantPack variantPack;
  std::vector<const phi::DenseTensor *> inputs = {input_x_tensor};
  std::vector<const phi::DenseTensor *> outputs = {out_tensor.get()};
  BuildVariantPack(inputs, outputs, variantPack);

  AsdOps::Status st = g_sortPlan->Setup(handle, variantPack);
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External("sortPlan Setup plan failed,"
                                          "ret message: %s .",
                                          st.Message()));
  variantPack.workspaceSize = g_sortPlan->GetWorkspaceSize();
  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }
  st = g_sortPlan->Execute(handle, variantPack);
  PADDLE_ENFORCE_EQ(
      st.Ok(),
      true,
      phi::errors::External("g_sortPlan Execute plan failed,"
                            "ret message: %s .",
                            st.Message()));
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(sort_op)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"num: int"})
    .SetKernelFn(PD_KERNEL(SortOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        SortOpInferShape));  // neccessary if the op has muti_inputs