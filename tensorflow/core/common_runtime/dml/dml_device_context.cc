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

    DmlDevice* dml_device = dynamic_cast<DmlDevice*>(device);
    if (!dml_device) {
      done(errors::Internal("Device should be DML, but is: ",
                            device->DebugString()));
      return;
    }

    ComPtr<ID3D12Device> d3d12_device = dml_device->GetD3D12Device();

    UINT64 width = dst_resource->GetDesc().Width;
    ComPtr<ID3D12Resource> upload_buffer;
    THROW_IF_FAILED(d3d12_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(width),
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&upload_buffer)));

    MapAndCopyToResource(upload_buffer.Get(), src_data, total_bytes);

    dml_device->AddCopyOperation(dst_resource.Get(), upload_buffer.Get());
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

    DmlDevice* dml_device = dynamic_cast<DmlDevice*>(device);
    if (!dml_device) {
      done(errors::Internal("Device should be DML, but is: ",
                            device->DebugString()));
      return;
    }
    ComPtr<ID3D12Device> d3d12_device = dml_device->GetD3D12Device();
    dml_device->AwaitComputeExecution();

    UINT64 width = src_resource->GetDesc().Width;
    ComPtr<ID3D12Resource> readback_buffer;
    THROW_IF_FAILED(d3d12_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(width),
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback_buffer)));

    dml_device->AddCopyOperation(readback_buffer.Get(), src_resource.Get());
    dml_device->AwaitCopyExecution();

    MapCopyFromResource(readback_buffer.Get(), dst_data, total_bytes);
  }
  done(Status::OK());
}

void DmlDeviceContext::MapAndCopyToResource(ID3D12Resource* resource,
                                            const void* src, UINT64 num_bytes) {
  D3D12_RANGE range = {0, static_cast<SIZE_T>(num_bytes)};
  void* data;
  THROW_IF_FAILED(resource->Map(0, &range, reinterpret_cast<void**>(&data)));
  memcpy(data, src, static_cast<SIZE_T>(num_bytes));
  resource->Unmap(0, &range);
}

void DmlDeviceContext::MapCopyFromResource(ID3D12Resource* resource, void* dest,
                                           UINT64 num_bytes) {
  D3D12_RANGE range = {0, static_cast<SIZE_T>(num_bytes)};
  void* data;
  THROW_IF_FAILED(resource->Map(0, &range, reinterpret_cast<void**>(&data)));
  memcpy(dest, data, static_cast<SIZE_T>(num_bytes));
  range.End = 0;
  resource->Unmap(0, &range);
}

}  // namespace tensorflow
//#endif  // TENSORFLOW_USE_DML
