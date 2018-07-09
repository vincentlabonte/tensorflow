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

#ifndef TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include <D3d12.h>
#include <stdlib.h>
#include <wrl/client.h>
#include <dml.h>
#include "d3dx12.h"
#include "dml_util.h"

using Microsoft::WRL::ComPtr;

struct CD3DX12_RESOURCE_DESC;

namespace tensorflow {

class DmlAllocator : public Allocator {
 public:
  DmlAllocator(ID3D12Device* device, const D3D12_HEAP_PROPERTIES& heapProps,
               D3D12_HEAP_FLAGS heapFlags, D3D12_RESOURCE_FLAGS resourceFlags,
               D3D12_RESOURCE_STATES initialState);
  virtual ~DmlAllocator() override;
  string Name() override;
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  virtual bool ShouldAllocateEmptyTensors() override final { return true; }

  // Returns a weak pointer to the ID3D12Resource associated with an opaque
  // allocation handle returned by AllocateRaw.
  ID3D12Resource* DecodeDataHandle(const void* opaqueHandle);

 private:
  static const uint32_t c_minResourceSizeExponent = 16;  // 2^16 = 64KB

  // The pool consists of a number of buckets, and each bucket contains a number
  // of resources of the same size. The resources in each bucket are always
  // sized as a power of two, and each bucket contains resources twice as large
  // as the previous bucket.
  struct Bucket {
    mutex mu;
    std::vector<ComPtr<ID3D12Resource>> resources GUARDED_BY(mu);
  };

  static std::ptrdiff_t GetBucketIndexFromSize(uint64_t size);
  static uint64_t GetBucketSizeFromIndex(std::ptrdiff_t index);

  ComPtr<ID3D12Device> m_device;
  D3D12_HEAP_PROPERTIES m_heapProperties;
  D3D12_HEAP_FLAGS m_heapFlags;
  D3D12_RESOURCE_FLAGS m_resourceFlags;
  D3D12_RESOURCE_STATES m_initialState;

  std::vector<Bucket> m_pool;

  TF_DISALLOW_COPY_AND_ASSIGN(DmlAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_
