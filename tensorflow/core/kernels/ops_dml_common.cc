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

DML_TENSOR_DESC DmlUtil::CreateDmlTensorDesc(const Tensor* tensor) {
  if (tensor->dtype() != DataType::DT_FLOAT) throw E_INVALIDARG;
  int dims = tensor->dims();
  if (dims > DML_TENSOR_DIMENSION_COUNT_NCHW) throw E_INVALIDARG;
  DML_TENSOR_DESC dml_tensor_desc = {DML_TENSOR_DATA_TYPE_FLOAT32,
                                     DML_TENSOR_FLAGS_NONE,
                                     DML_TENSOR_DIMENSION_COUNT_NCHW,
                                     {1, 1, 1, 1}};
  auto dim_sizes = tensor->shape().dim_sizes();
  for (int i = 0; i < dims; i++) {
    dml_tensor_desc.sizes[DML_TENSOR_DIMENSION_COUNT_NCHW - 1 - i] =
        dim_sizes[dims - 1 - i];
  }
  return dml_tensor_desc;
}

DML_TENSOR_DESC DmlUtil::CreateDmlTensorDesc(const Tensor* tensor,
                                             const Tensor* other_tensor) {
  if (tensor->dtype() != DataType::DT_FLOAT) throw E_INVALIDARG;
  int dims = tensor->dims();
  int other_dims = other_tensor->dims();
  int max_dims = std::max(dims, other_dims);
  if (dims > DML_TENSOR_DIMENSION_COUNT_NCHW) throw E_INVALIDARG;
  DML_TENSOR_DESC dml_tensor_desc = {DML_TENSOR_DATA_TYPE_FLOAT32,
                                     DML_TENSOR_FLAGS_USE_STRIDES,
                                     DML_TENSOR_DIMENSION_COUNT_NCHW,
                                     {1, 1, 1, 1}};
  auto dim_sizes = tensor->shape().dim_sizes();
  auto other_dim_sizes = other_tensor->shape().dim_sizes();
  UINT stride_value = 1u;
  for (int i = max_dims - 1; i >= 0; i--) {
    if (i >= max_dims - dims && i >= max_dims - other_dims) {
      int64 max_dim_size = std::max(dim_sizes[i], other_dim_sizes[i]);
      if (dim_sizes[i] == 1) {
        dml_tensor_desc.strides[i] = 0;
      } else if (dim_sizes[i] == max_dim_size) {
        dml_tensor_desc.strides[i] = stride_value;
      } else {
        throw E_INVALIDARG;
      }
      dml_tensor_desc.sizes[i] = max_dim_size;
      stride_value *= max_dim_size;
    } else if (i >= max_dims - other_dims) {
      dml_tensor_desc.strides[i] = 0;
      dml_tensor_desc.sizes[i] = other_dim_sizes[i];
      stride_value *= other_dim_sizes[i];
    } else if (i >= max_dims - dims) {
      dml_tensor_desc.strides[i] = stride_value;
      dml_tensor_desc.sizes[i] = dim_sizes[i];
      stride_value *= dim_sizes[i];
    } else {
      throw E_INVALIDARG;
    }
  }
  return dml_tensor_desc;
}

void DmlBinaryOp::Compute(OpKernelContext* ctx) {
  ComPtr<IDXGraphicsAnalysis> ga;
  HRESULT hr = DXGIGetDebugInterface1(0, IID_PPV_ARGS(&ga));
  if (SUCCEEDED(hr)) {
    ga->BeginCapture();
  }

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

  DmlInterface* dml_interface = DmlInterface::instance();
  ComPtr<IDMLDevice> dml_device = dml_interface->GetDmlDevice();
  ComPtr<IDMLDeviceContext> dml_device_context;
  THROW_IF_FAILED(dml_device->CreateDeviceContext(
      dml_interface->GetD3D12Fence(), &dml_device_context));

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

  THROW_IF_FAILED(dml_device_context->Open(dml_interface->GetFenceValue() + 1));

  IDMLResource* input_resources[2] = {in0_dml_resource.Get(),
                                      in1_dml_resource.Get()};
  THROW_IF_FAILED(
      dml_device_context->AddOperation(dml_operation.Get(), input_resources, 2,
                                       out_dml_resource.GetAddressOf(), 1));

  ComPtr<ID3D12CommandList> compute_command_list;
  THROW_IF_FAILED(dml_device_context->Close(&compute_command_list));

  ID3D12CommandList* compute_command_lists[1] = {compute_command_list.Get()};

  dml_interface->GetD3D12CommandQueue()->ExecuteCommandLists(
      1, compute_command_lists);

  dml_interface->AwaitExecution();

  if (SUCCEEDED(hr)) {
    ga->EndCapture();
  }
}

void DmlActivationOp::Compute(OpKernelContext* ctx) {
  ComPtr<IDXGraphicsAnalysis> ga;
  HRESULT hr = DXGIGetDebugInterface1(0, IID_PPV_ARGS(&ga));
  if (SUCCEEDED(hr)) {
    ga->BeginCapture();
  }

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

  DmlInterface* dml_interface = DmlInterface::instance();
  ComPtr<IDMLDevice> dml_device = dml_interface->GetDmlDevice();
  ComPtr<IDMLDeviceContext> dml_device_context;
  THROW_IF_FAILED(dml_device->CreateDeviceContext(
      dml_interface->GetD3D12Fence(), &dml_device_context));

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

  THROW_IF_FAILED(dml_device_context->Open(dml_interface->GetFenceValue() + 1));

  IDMLResource* input_resources[1] = {input_dml_resource.Get()};
  THROW_IF_FAILED(
      dml_device_context->AddOperation(dml_operation.Get(), input_resources, 1,
                                       output_dml_resource.GetAddressOf(), 1));

  ComPtr<ID3D12CommandList> compute_command_list;
  THROW_IF_FAILED(dml_device_context->Close(&compute_command_list));

  ID3D12CommandList* compute_command_lists[1] = {compute_command_list.Get()};

  dml_interface->GetD3D12CommandQueue()->ExecuteCommandLists(
      1, compute_command_lists);

  dml_interface->AwaitExecution();

  if (SUCCEEDED(hr)) {
    ga->EndCapture();
  }
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
