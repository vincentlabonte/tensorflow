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

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/dml/dml_allocator.h"
#include "tensorflow/core/common_runtime/dml/dml_device_context.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class DmlInterface {
  Allocator* m_cpu_allocator_;       // not owned
  DmlAllocator* m_dml_allocator_;    // owned
  DmlDeviceContext* m_dml_context_;  // ref counted
  DmlInterface() {
    m_cpu_allocator_ = cpu_allocator();
    m_dml_allocator_ = new DmlAllocator();
    m_dml_context_ = new DmlDeviceContext();
  }

  ~DmlInterface() {
    delete m_dml_allocator_;
    m_dml_context_->Unref();
  }

  void AddDevice() {}

 public:
  static const DmlInterface* instance() {
    // c++11 guarantees that this will be constructed in a thread safe way
    static const DmlInterface instance;
    return &instance;
  }

  DmlAllocator* GetDmlAllocator() const { return m_dml_allocator_; }

  Allocator* GetCPUAllocator() const { return m_cpu_allocator_; }

  DmlDeviceContext* GetDmlContext() const { return m_dml_context_; }
};

class DmlDevice : public LocalDevice {
 public:
  DmlDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            const string& physical_device_desc, DmlAllocator* dml_allocator,
            Allocator* cpu_allocator, DmlDeviceContext* ctx)
      : LocalDevice(options, Device::BuildDeviceAttributes(
                                 name, DEVICE_DML, memory_limit, locality,
                                 physical_device_desc)),
        cpu_allocator_(cpu_allocator),
        dml_allocator_(dml_allocator),
        device_context_(ctx) {}

  ~DmlDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map) override;

  Status Sync() override;

 private:
  Allocator* cpu_allocator_;          // not owned
  DmlAllocator* dml_allocator_;       // not owned
  DmlDeviceContext* device_context_;  // not owned
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_DEVICE_H_
