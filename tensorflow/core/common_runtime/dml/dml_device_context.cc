/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

//#if TENSORFLOW_USE_DML

#include "tensorflow/core/common_runtime/dml/dml_device_context.h"
#include "tensorflow/core/common_runtime/dml/dml_interface.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

void DmlDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  const int64 total_bytes = device_tensor->TotalBytes();
  // Tensors must be the same size
  assert(total_bytes == cpu_tensor->TotalBytes());

  if (total_bytes > 0) {
    const void* src_data = DMAHelper::base(cpu_tensor);
    const void* dst_data = DMAHelper::base(device_tensor);

    AllocatorAttributes attrs;
    DmlAllocator* allocator =
        static_cast<DmlAllocator*>(device->GetAllocator(attrs));
    ComPtr<ID3D12Resource> dst_resource = allocator->DecodeDataHandle(dst_data);

	DmlInterface* dml_interface = DmlInterface::instance();
    ComPtr<ID3D12Device> d3d12_device = dml_interface->GetD3D12Device();

    UINT64 width = dst_resource->GetDesc().Width;
    ComPtr<ID3D12Resource> uploadBuffer;
    THROW_IF_FAILED(d3d12_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(width),
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&uploadBuffer)));

    DmlInterface::MapAndCopyToResource(uploadBuffer.Get(), src_data,
                                       total_bytes);

	ComPtr<ID3D12CommandAllocator> pCommandAllocatorCopyToGPU;
    THROW_IF_FAILED(d3d12_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(&pCommandAllocatorCopyToGPU)));
    ComPtr<ID3D12GraphicsCommandList> pCommandListCopyToGPU;
    THROW_IF_FAILED(d3d12_device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_COMPUTE, pCommandAllocatorCopyToGPU.Get(),
        nullptr, IID_PPV_ARGS(&pCommandListCopyToGPU)));

    pCommandListCopyToGPU->CopyResource(dst_resource.Get(), uploadBuffer.Get());
    pCommandListCopyToGPU->Close();

    ID3D12CommandList* pCopyToGPUCLs[1] = {pCommandListCopyToGPU.Get()};
    dml_interface->GetD3D12CommandQueue()->ExecuteCommandLists(1, pCopyToGPUCLs);

    dml_interface->AwaitExecution();
  }
  done(Status::OK());
}

void DmlDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             StringPiece edge_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  const int64 total_bytes = device_tensor->TotalBytes();
  // Tensors must be the same size
  assert(total_bytes == cpu_tensor->TotalBytes());

  if (total_bytes > 0) {
    const void* src_data = DMAHelper::base(device_tensor);
    void* dst_data = DMAHelper::base(cpu_tensor);

    AllocatorAttributes attrs;
    DmlAllocator* allocator =
        static_cast<DmlAllocator*>(device->GetAllocator(attrs));
    ComPtr<ID3D12Resource> src_resource = allocator->DecodeDataHandle(src_data);

    DmlInterface* dml_interface = DmlInterface::instance();
    ComPtr<ID3D12Device> d3d12_device = dml_interface->GetD3D12Device();

	UINT64 width = src_resource->GetDesc().Width;
    ComPtr<ID3D12Resource> readbackBuffer;
    THROW_IF_FAILED(d3d12_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(width),
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readbackBuffer)));

	ComPtr<ID3D12CommandAllocator> pCommandAllocatorCopyFromGPU;
    THROW_IF_FAILED(d3d12_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(&pCommandAllocatorCopyFromGPU)));
    ComPtr<ID3D12GraphicsCommandList> pCommandListCopyFromGPU;
    THROW_IF_FAILED(d3d12_device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_COMPUTE, pCommandAllocatorCopyFromGPU.Get(),
        nullptr, IID_PPV_ARGS(&pCommandListCopyFromGPU)));

    pCommandListCopyFromGPU->CopyResource(readbackBuffer.Get(),
                                          src_resource.Get());
    pCommandListCopyFromGPU->Close();

    ID3D12CommandList* pCopyFromGPUCLs[1] = {pCommandListCopyFromGPU.Get()};
    dml_interface->GetD3D12CommandQueue()->ExecuteCommandLists(1,
                                                               pCopyFromGPUCLs);

    dml_interface->AwaitExecution();

    DmlInterface::MapCopyFromResource(readbackBuffer.Get(), dst_data,
                                      total_bytes);
  }
  done(Status::OK());
}

}  // namespace tensorflow
//#endif  // TENSORFLOW_USE_DML
