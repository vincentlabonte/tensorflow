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
#include <dml.h>
#include <windows.h>
#include <wrl/client.h>
#include <dxgi1_5.h>
#include <DXProgrammableCapture.h>

#include "tensorflow/core/common_runtime/dml/dml_allocator.h"
#include "tensorflow/core/common_runtime/dml/dml_interface.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"

using Microsoft::WRL::ComPtr;

namespace tensorflow {
REGISTER6(BinaryOp, CPU, "Add", functor::add, float, Eigen::half, double, int32,
          int64, bfloat16);
REGISTER6(BinaryOp, CPU, "AddV2", functor::add, float, Eigen::half, double,
          int32, int64, bfloat16);

#if GOOGLE_CUDA
REGISTER3(BinaryOp, GPU, "Add", functor::add, float, Eigen::half, double);
REGISTER3(BinaryOp, GPU, "AddV2", functor::add, float, Eigen::half, double);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Add")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
REGISTER_KERNEL_BUILDER(Name("AddV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
#endif

#if TENSORFLOW_USE_SYCL
#define REGISTER_KERNEL(type)                          \
  REGISTER(BinaryOp, SYCL, "Add", functor::add, type); \
  REEGISTER(BinaryOp, SYCL, "AddV2", functor::add, type);

TF_CALL_SYCL_NUMBER_TYPES(REGISTER_KERNEL);

REGISTER_KERNEL_BUILDER(Name("Add")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
REGISTER_KERNEL_BUILDER(Name("AddV2")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
#endif  // TENSORFLOW_USE_SYCL

class DmlAddBinaryOp : public BinaryOpShared {
 public:
  explicit DmlAddBinaryOp(OpKernelConstruction* ctx)
      : BinaryOpShared(ctx, DataTypeToEnum<float>::v(),
                       DataTypeToEnum<float>::v()) {}

  void Compute(OpKernelContext* ctx) override {
    ComPtr<IDXGraphicsAnalysis> ga;
    HRESULT hr = DXGIGetDebugInterface1(0, IID_PPV_ARGS(&ga));
    if (hr != E_NOINTERFACE) {
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

    AllocatorAttributes attrs;
    DmlAllocator* allocator =
        static_cast<DmlAllocator*>(ctx->device()->GetAllocator(attrs));
    // out = new Tensor(allocator, DataType::DT_FLOAT, in0.shape());

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
        {DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_FLAGS_NONE, 4, {1, 1, 1, 5}},
        {DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_FLAGS_NONE, 4, {1, 1, 1, 5}}};

    DML_TENSOR_DESC const* dml_input_ref[2] = {&dml_input_desc[0],
                                               &dml_input_desc[1]};

    DML_TENSOR_DESC dml_output_desc = {
        DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_FLAGS_NONE, 4, {1, 1, 1, 5}};

    ComPtr<IDMLOperation> dml_operation;
    THROW_IF_FAILED(dml_device->CreateElementWiseOperation(
        DML_ELEMENT_WISE_FUNCTION_ADD, &dml_input_ref[0], 2, &dml_output_desc,
        nullptr,  // params
        DML_EXECUTION_HINT_FLAGS_NONE, &dml_operation));

    THROW_IF_FAILED(
        dml_device_context->Open(dml_interface->GetFenceValue() + 1));

    IDMLResource* input_resources[2] = {in0_dml_resource.Get(),
                                        in1_dml_resource.Get()};
    THROW_IF_FAILED(dml_device_context->AddOperation(
        dml_operation.Get(), input_resources, 2,
        out_dml_resource.GetAddressOf(), 1));

    ComPtr<ID3D12CommandList> compute_command_list;
    THROW_IF_FAILED(dml_device_context->Close(&compute_command_list));

    ID3D12CommandList* compute_command_lists[1] = {compute_command_list.Get()};

    dml_interface->GetD3D12CommandQueue()->ExecuteCommandLists(
        1, compute_command_lists);

    dml_interface->AwaitExecution();

    if (hr != E_NOINTERFACE) {
      ga->EndCapture();
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Add").Device(DEVICE_DML).TypeConstraint<float>("T"), DmlAddBinaryOp);

}  // namespace tensorflow
