/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/ops_dml_common.h"

namespace tensorflow {

void DmlBinaryOp::Compute(OpKernelContext* ctx) {
  BinaryOpState state(ctx);
  if (!ctx->status().ok()) return;
  Tensor* out = state.out;
  BCast* bcast = &state.bcast;
  auto& in0 = state.in0;
  auto& in1 = state.in1;
  if (state.out_num_elements == 0) {
    return;
  }

  const int ndims = state.ndims;
  if (ndims > DML_TENSOR_DIMENSION_COUNT_NCHW) {
    return;
  }

  AllocatorAttributes attrs;
  DmlAllocator* allocator =
      static_cast<DmlAllocator*>(ctx->device()->GetAllocator(attrs));

  const void* in0_data = in0.tensor_data().data();
  const void* in1_data = in1.tensor_data().data();
  const void* out_data = out->tensor_data().data();

  ComPtr<ID3D12Resource> in0_resource = allocator->DecodeDataHandle(in0_data);
  ComPtr<ID3D12Resource> in1_resource = allocator->DecodeDataHandle(in1_data);
  ComPtr<ID3D12Resource> out_resource = allocator->DecodeDataHandle(out_data);

  DmlDevice* device = dynamic_cast<DmlDevice*>(ctx->device());
  OP_REQUIRES(ctx, device,
              errors::Internal("Device should be DML, but is: ",
                               ctx->device()->name()));
  ComPtr<IDMLDevice> dml_device = device->GetDmlDevice();
  ComPtr<IDMLDeviceContext> dml_device_context = device->GetDmlDeviceContext();

  ComPtr<IDMLResource> in0_dml_resource;
  ComPtr<IDMLResource> in1_dml_resource;
  ComPtr<IDMLResource> out_dml_resource;

  THROW_IF_FAILED(dml_device_context->CreateResource(in0_resource.Get(),
                                                     &in0_dml_resource));
  THROW_IF_FAILED(dml_device_context->CreateResource(in1_resource.Get(),
                                                     &in1_dml_resource));
  THROW_IF_FAILED(dml_device_context->CreateResource(out_resource.Get(),
                                                     &out_dml_resource));

  DML_TENSOR_DESC dml_input_desc[2] = {
      DmlUtil::CreateDmlTensorDesc(&in0, &in1),
      DmlUtil::CreateDmlTensorDesc(&in1, &in0)};

  DML_TENSOR_DESC const* dml_input_ref[2] = {&dml_input_desc[0],
                                             &dml_input_desc[1]};

  DML_TENSOR_DESC dml_output_desc = {DmlUtil::CreateDmlTensorDesc(out)};

  ComPtr<IDMLOperation> dml_operation;
  THROW_IF_FAILED(dml_device->CreateElementWiseOperation(
      GetDmlElementWiseFunction(), &dml_input_ref[0], 2, &dml_output_desc,
      nullptr,  // params
      DML_EXECUTION_HINT_FLAGS_NONE, &dml_operation));

  IDMLResource* input_resources[2] = {in0_dml_resource.Get(),
                                      in1_dml_resource.Get()};
  THROW_IF_FAILED(
      device->AddComputeOperation(dml_operation.Get(), input_resources, 2,
                                  out_dml_resource.GetAddressOf(), 1));
}

void DmlActivationOp::Compute(OpKernelContext* ctx) {
  const Tensor& input = ctx->input(0);
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, input.shape(), &output));

  if (!ctx->status().ok()) return;
  if (input.NumElements() <= 0) return;
  if (input.dims() > DML_TENSOR_DIMENSION_COUNT_NCHW) return;

  AllocatorAttributes attrs;
  DmlAllocator* allocator =
      static_cast<DmlAllocator*>(ctx->device()->GetAllocator(attrs));

  const void* input_data = input.tensor_data().data();
  const void* output_data = output->tensor_data().data();

  ComPtr<ID3D12Resource> input_resource =
      allocator->DecodeDataHandle(input_data);
  ComPtr<ID3D12Resource> output_resource =
      allocator->DecodeDataHandle(output_data);

  DmlDevice* device = dynamic_cast<DmlDevice*>(ctx->device());
  OP_REQUIRES(ctx, device,
              errors::Internal("Device should be DML, but is: ",
                               ctx->device()->name()));
  ComPtr<IDMLDevice> dml_device = device->GetDmlDevice();
  ComPtr<IDMLDeviceContext> dml_device_context = device->GetDmlDeviceContext();

  ComPtr<IDMLResource> input_dml_resource;
  ComPtr<IDMLResource> output_dml_resource;

  THROW_IF_FAILED(dml_device_context->CreateResource(input_resource.Get(),
                                                     &input_dml_resource));
  THROW_IF_FAILED(dml_device_context->CreateResource(output_resource.Get(),
                                                     &output_dml_resource));

  const DML_TENSOR_DESC dml_input_desc = DmlUtil::CreateDmlTensorDesc(&input);
  const DML_TENSOR_DESC dml_output_desc = DmlUtil::CreateDmlTensorDesc(output);

  ComPtr<IDMLOperation> dml_operation;
  THROW_IF_FAILED(dml_device->CreateActivationOperation(
      GetDmlActivationFunction(), &dml_input_desc, nullptr, &dml_output_desc,
      nullptr, DML_EXECUTION_HINT_FLAGS_NONE, &dml_operation));

  IDMLResource* input_resources[1] = {input_dml_resource.Get()};
  THROW_IF_FAILED(
      device->AddComputeOperation(dml_operation.Get(), input_resources, 1,
                                  output_dml_resource.GetAddressOf(), 1));
}

class DmlReluOp : public DmlActivationOp {
 public:
  explicit DmlReluOp(OpKernelConstruction* ctx) : DmlActivationOp(ctx) {}

  DML_ACTIVATION_FUNCTION GetDmlActivationFunction() override {
    return DML_ACTIVATION_FUNCTION_RELU;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device(DEVICE_DML).TypeConstraint<float>("T"), DmlReluOp);

}  // namespace tensorflow
