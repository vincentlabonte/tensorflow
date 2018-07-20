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
  Allocator* cpu_allocator_;          // not owned
  DmlAllocator* dml_allocator_;       // owned
  DmlDeviceContext* device_context_;  // ref counted

  ComPtr<ID3D12Device> d3d12_device_;
  ComPtr<IDMLDevice> dml_device_;

  ComPtr<ID3D12CommandQueue> compute_command_queue_;
  ComPtr<ID3D12CommandQueue> copy_command_queue_;

  ComPtr<ID3D12Fence> compute_fence_;
  ComPtr<ID3D12Fence> copy_fence_;

  mutex compute_fence_value_mu_;
  uint64_t compute_fence_value_ = 0 GUARDED_BY(compute_fence_value_mu_);

  mutex copy_fence_value_mu_;
  uint64_t copy_fence_value_ = 0 GUARDED_BY(copy_fence_value_mu_);

  mutex dml_device_context_mu_;
  ComPtr<IDMLDeviceContext> dml_device_context_
      GUARDED_BY(dml_device_context_mu_);
  bool is_dml_device_context_open_ GUARDED_BY(dml_device_context_mu_);

  DmlInterface() : is_dml_device_context_open_(false) {
    THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                      IID_PPV_ARGS(&d3d12_device_)));
    THROW_IF_FAILED(
        DMLCreateDevice(d3d12_device_.Get(), IID_PPV_ARGS(&dml_device_)));

    D3D12_COMMAND_QUEUE_DESC compute_command_queue_desc = {
        D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    THROW_IF_FAILED(d3d12_device_->CreateCommandQueue(
        &compute_command_queue_desc, IID_PPV_ARGS(&compute_command_queue_)));

    D3D12_COMMAND_QUEUE_DESC copy_command_queue_desc = {
        D3D12_COMMAND_LIST_TYPE_COPY, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    THROW_IF_FAILED(d3d12_device_->CreateCommandQueue(
        &copy_command_queue_desc, IID_PPV_ARGS(&copy_command_queue_)));

    cpu_allocator_ = cpu_allocator();
    dml_allocator_ = new DmlAllocator(
        d3d12_device_.Get(), CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON);
    device_context_ = new DmlDeviceContext();

    THROW_IF_FAILED(d3d12_device_.Get()->CreateFence(
        compute_fence_value_, D3D12_FENCE_FLAG_SHARED,
        IID_PPV_ARGS(&compute_fence_)));
    THROW_IF_FAILED(d3d12_device_.Get()->CreateFence(
        copy_fence_value_, D3D12_FENCE_FLAG_SHARED,
        IID_PPV_ARGS(&copy_fence_)));

    THROW_IF_FAILED(dml_device_->CreateDeviceContext(compute_fence_.Get(),
                                                     &dml_device_context_));
  }

  ~DmlInterface() {
    delete dml_allocator_;
    device_context_->Unref();
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

  DmlAllocator* GetDmlAllocator() const { return dml_allocator_; }

  Allocator* GetCPUAllocator() const { return cpu_allocator_; }

  DmlDeviceContext* GetDeviceContext() const { return device_context_; }

  IDMLDeviceContext* GetDmlDeviceContext() const {
    return dml_device_context_.Get();
  }

  IDMLDevice* GetDmlDevice() const { return dml_device_.Get(); }

  ID3D12Device* GetD3D12Device() const { return d3d12_device_.Get(); }

  ID3D12CommandQueue* GetComputeCommandQueue() const {
    return compute_command_queue_.Get();
  }

  ID3D12CommandQueue* GetCopyCommandQueue() const {
    return copy_command_queue_.Get();
  }

  ID3D12Fence* GetComputeFence() const { return compute_fence_.Get(); }

  ID3D12Fence* GetCopyFence() const { return copy_fence_.Get(); }

  HRESULT AddComputeOperation(IDMLOperation* operation,
                              IDMLResource* const* input_resources,
                              UINT input_count,
                              IDMLResource* const* output_resources,
                              UINT output_count) {
    dml_device_context_mu_.lock();
    if (!is_dml_device_context_open_) {
      compute_fence_value_mu_.lock();
      dml_device_context_->Open(compute_fence_value_ + 1);
      compute_fence_value_mu_.unlock();
      is_dml_device_context_open_ = true;
    }
    HRESULT ret = dml_device_context_->AddOperation(
        operation, input_resources, input_count, output_resources,
        output_count);
    dml_device_context_mu_.unlock();
    return ret;
  }

  void AwaitComputeExecution() {
    dml_device_context_mu_.lock();
    if (is_dml_device_context_open_) {
      ComPtr<ID3D12CommandList> compute_command_list;
      THROW_IF_FAILED(dml_device_context_->Close(&compute_command_list));
      ID3D12CommandList* compute_command_lists[1] = {
          compute_command_list.Get()};
      compute_command_queue_->ExecuteCommandLists(1, compute_command_lists);
      is_dml_device_context_open_ = false;
    }
    dml_device_context_mu_.unlock();

    HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    compute_fence_value_mu_.lock();
    ++compute_fence_value_;
    THROW_IF_FAILED(compute_command_queue_->Signal(compute_fence_.Get(),
                                                   compute_fence_value_));
    THROW_IF_FAILED(compute_fence_->SetEventOnCompletion(compute_fence_value_,
                                                         fence_event));
    compute_fence_value_mu_.unlock();

    DWORD retVal = WaitForSingleObject(fence_event, INFINITE);
    if (retVal != WAIT_OBJECT_0) {
      DebugBreak();
    }
  }

  void AwaitCopyExecution() {
    HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    copy_fence_value_mu_.lock();
    ++copy_fence_value_;
    THROW_IF_FAILED(
        copy_command_queue_->Signal(copy_fence_.Get(), copy_fence_value_));
    THROW_IF_FAILED(
        copy_fence_->SetEventOnCompletion(copy_fence_value_, fence_event));
    copy_fence_value_mu_.unlock();

    DWORD retVal = WaitForSingleObject(fence_event, INFINITE);
    if (retVal != WAIT_OBJECT_0) {
      DebugBreak();
    }
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_DEVICE_H_