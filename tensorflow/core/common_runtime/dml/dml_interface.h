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

//#if !TENSORFLOW_USE_DML
//#error This file must only be included when building TensorFlow with DML
// support #endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_INTERFACE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_INTERFACE_H_

#include "tensorflow/core/common_runtime/dml/dml_allocator.h"
#include "tensorflow/core/common_runtime/dml/dml_device_context.h"

#include <Objbase.h>

namespace tensorflow {

class DmlInterface {
  Allocator* m_cpu_allocator_;              // not owned
  DmlAllocator* m_dml_allocator_;           // owned
  DmlDeviceContext* m_dml_device_context_;  // ref counted

  ComPtr<ID3D12Device> m_d3d12_device_;
  ComPtr<IDMLDevice> m_dml_device_;
  ComPtr<ID3D12CommandQueue> m_d3d12_command_queue_;
  ComPtr<ID3D12Fence> m_d3d12_fence_;
  HANDLE event;
  uint64_t fence_value = 0;

  DmlInterface() {
    THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                      IID_PPV_ARGS(&m_d3d12_device_)));
    THROW_IF_FAILED(
        DMLCreateDevice(m_d3d12_device_.Get(), IID_PPV_ARGS(&m_dml_device_)));

    D3D12_COMMAND_QUEUE_DESC command_queue_desc = {
        D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    THROW_IF_FAILED(m_d3d12_device_->CreateCommandQueue(
        &command_queue_desc, IID_PPV_ARGS(&m_d3d12_command_queue_)));

    m_cpu_allocator_ = cpu_allocator();
    m_dml_allocator_ = new DmlAllocator(
        m_d3d12_device_.Get(), CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON);
    m_dml_device_context_ = new DmlDeviceContext();

    event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    THROW_IF_FAILED(m_d3d12_device_.Get()->CreateFence(
        fence_value, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_d3d12_fence_)));
  }

  ~DmlInterface() {
    delete m_dml_allocator_;
    m_dml_device_context_->Unref();
  }

 public:
  static DmlInterface* instance() {
    // c++11 guarantees that this will be constructed in a thread safe way
    static DmlInterface instance;
    return &instance;
  }

  static void MapAndCopyToResource(ID3D12Resource* pResource, const void* pSrc,
                                   UINT64 numBytes) {
    D3D12_RANGE range = {0, static_cast<SIZE_T>(numBytes)};
    void* pData;
    THROW_IF_FAILED(
        pResource->Map(0, &range, reinterpret_cast<void**>(&pData)));
    memcpy(pData, pSrc, static_cast<SIZE_T>(numBytes));
    pResource->Unmap(0, &range);
  }

  static void MapCopyFromResource(ID3D12Resource* pResource, void* pDest,
                                  UINT64 numBytes) {
    D3D12_RANGE range = {0, static_cast<SIZE_T>(numBytes)};
    void* pData;
    THROW_IF_FAILED(
        pResource->Map(0, &range, reinterpret_cast<void**>(&pData)));
    memcpy(pDest, pData, static_cast<SIZE_T>(numBytes));
    range.End = 0;
    pResource->Unmap(0, &range);
  }

  DmlAllocator* GetDmlAllocator() const { return m_dml_allocator_; }

  Allocator* GetCPUAllocator() const { return m_cpu_allocator_; }

  DmlDeviceContext* GetDmlDeviceContext() const {
    return m_dml_device_context_;
  }

  IDMLDevice* GetDmlDevice() const { return m_dml_device_.Get(); }

  ID3D12Device* GetD3D12Device() const { return m_d3d12_device_.Get(); }

  ID3D12CommandQueue* GetD3D12CommandQueue() const {
    return m_d3d12_command_queue_.Get();
  }

  ID3D12Fence* GetD3D12Fence() const { return m_d3d12_fence_.Get(); }

  uint64_t GetFenceValue() const { return fence_value; }

  void AwaitExecution() {
    ++fence_value;
    THROW_IF_FAILED(
        m_d3d12_command_queue_->Signal(m_d3d12_fence_.Get(), fence_value));

    THROW_IF_FAILED(m_d3d12_fence_->SetEventOnCompletion(fence_value, event));

    DWORD retVal = WaitForSingleObject(event, INFINITE);
    if (retVal != WAIT_OBJECT_0) {
      DebugBreak();
    }
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_DEVICE_H_