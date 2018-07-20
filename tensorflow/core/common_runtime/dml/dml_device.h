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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_DEVICE_H_

#include "tensorflow/core/common_runtime/dml/dml_allocator.h"
#include "tensorflow/core/common_runtime/dml/dml_device_context.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/public/session_options.h"

#include <Objbase.h>
#include <dml.h>

namespace tensorflow {

class DmlDeviceContext;

class DmlDevice : public LocalDevice {
 public:
  DmlDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            Allocator* cpu_allocator);

  ~DmlDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map) override;

  Status Sync() override;

  ID3D12Device* GetD3D12Device() const;
  IDMLDevice* GetDmlDevice() const;
  IDMLDeviceContext* GetDmlDeviceContext() const;
  ID3D12CommandQueue* GetCopyCommandQueue() const;
  ID3D12CommandAllocator* GetCopyCommandAllocator() const;

  HRESULT AddComputeOperation(IDMLOperation* operation,
                              IDMLResource* const* input_resources,
                              UINT input_count,
                              IDMLResource* const* output_resources,
                              UINT output_count);

  void AwaitComputeExecution();

  void AwaitCopyExecution();

 private:
  // TensorFlow
  Allocator* cpu_allocator_;
  DmlAllocator* dml_allocator_;
  DmlDeviceContext* device_context_;

  // DML
  ComPtr<ID3D12Device> d3d12_device_;

  // Compute
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<ID3D12CommandQueue> compute_command_queue_;
  mutex dml_device_context_mu_;
  ComPtr<IDMLDeviceContext> dml_device_context_
      GUARDED_BY(dml_device_context_mu_);
  bool is_dml_device_context_open_ GUARDED_BY(dml_device_context_mu_);
  ComPtr<ID3D12Fence> compute_fence_;
  mutex compute_fence_value_mu_;
  uint64_t compute_fence_value_ GUARDED_BY(compute_fence_value_mu_);

  // Copy
  ComPtr<ID3D12CommandQueue> copy_command_queue_;
  ComPtr<ID3D12CommandAllocator> copy_command_allocator_;
  ComPtr<ID3D12Fence> copy_fence_;
  mutex copy_fence_value_mu_;
  uint64_t copy_fence_value_ GUARDED_BY(copy_fence_value_mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_DEVICE_H_
