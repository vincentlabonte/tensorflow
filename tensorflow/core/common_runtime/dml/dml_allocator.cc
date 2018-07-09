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

//#ifdef TENSORFLOW_USE_DML

#include "tensorflow/core/common_runtime/dml/dml_allocator.h"

namespace tensorflow {

struct AllocationInfo {
  DmlAllocator* owner = nullptr;
  ComPtr<ID3D12Resource> resource;

  // The size requested during Alloc(), which may be smaller than the physical
  // resource size
  size_t requestedSize = 0;
};

DmlAllocator::DmlAllocator(ID3D12Device* device,
                           const D3D12_HEAP_PROPERTIES& heapProps,
                           D3D12_HEAP_FLAGS heapFlags,
                           D3D12_RESOURCE_FLAGS resourceFlags,
                           D3D12_RESOURCE_STATES initialState)
    : m_device(device),
      m_heapProperties(heapProps),
      m_heapFlags(heapFlags),
      m_resourceFlags(resourceFlags),
      m_initialState(initialState) {}

DmlAllocator::~DmlAllocator() {}

/*static*/ std::ptrdiff_t DmlAllocator::GetBucketIndexFromSize(uint64_t size) {
  assert(size != 0);

  // Each bucket is twice as large as the previous one, in ascending order
  std::ptrdiff_t index = static_cast<std::ptrdiff_t>(ceil(log2(size)));
  assert((1ull << index) >= size);  // This must be true unless there were some
                                    // strange rounding issues

  // The smallest bucket is 2^n bytes large, where n = c_minResourceSizeExponent
  index = std::max<std::ptrdiff_t>(index, c_minResourceSizeExponent);
  index -= c_minResourceSizeExponent;

  return index;
}

/*static*/ uint64_t DmlAllocator::GetBucketSizeFromIndex(std::ptrdiff_t index) {
  return (1ull << (index + c_minResourceSizeExponent));
}

string DmlAllocator::Name() { return "device:DML"; }

void* DmlAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  size_t size = std::max<size_t>(1, num_bytes);

  // Find the bucket for this allocation size
  std::ptrdiff_t bucketIndex = GetBucketIndexFromSize(size);

  if (m_pool.size() <= bucketIndex) {
    // Ensure there are sufficient buckets
    m_pool.resize(bucketIndex + 1);
  }

  Bucket& bucket = m_pool[bucketIndex];
  const uint64_t bucketSize = GetBucketSizeFromIndex(bucketIndex);

  ComPtr<ID3D12Resource> resource;
  bucket.mu.lock();
  if (bucket.resources.empty()) {
    bucket.mu.unlock();
    // No more resources in this bucket - allocate a new one
    THROW_IF_FAILED(m_device->CreateCommittedResource(
        &m_heapProperties, m_heapFlags,
        &CD3DX12_RESOURCE_DESC::Buffer(bucketSize, m_resourceFlags),
        m_initialState, nullptr, IID_PPV_ARGS(&resource)));
  } else {
    // Retrieve a resource from the bucket
    resource = std::move(bucket.resources.back());
    bucket.resources.pop_back();
    bucket.mu.unlock();
  }

  assert(resource != nullptr);
  assert(resource->GetDesc().Width == bucketSize);

  auto allocInfo = std::make_unique<AllocationInfo>();
  allocInfo->owner = this;
  allocInfo->resource = std::move(resource);  // Take ownership of this resource
  allocInfo->requestedSize = size;

  return allocInfo.release();  // "Detach"
}

void DmlAllocator::DeallocateRaw(void* ptr) {
  std::unique_ptr<AllocationInfo> allocInfo(static_cast<AllocationInfo*>(ptr));

  assert(allocInfo != nullptr);  // Can't free nullptr

  if (allocInfo->owner != this) {
    // This allocation doesn't belong to this allocator!
    allocInfo.release();
    throw E_INVALIDARG;
  }

  std::ptrdiff_t bucketIndex = GetBucketIndexFromSize(allocInfo->requestedSize);

  // The resource size must match the bucket size...
  assert(GetBucketSizeFromIndex(bucketIndex) ==
         allocInfo->resource->GetDesc().Width);
  assert(m_pool.size() > bucketIndex);

  // Return the resource to the bucket
  Bucket& bucket = m_pool[bucketIndex];
  bucket.mu.lock();
  bucket.resources.push_back(std::move(allocInfo->resource));
  bucket.mu.unlock();

  // Free the allocation info
  allocInfo.reset();
}

ID3D12Resource* DmlAllocator::DecodeDataHandle(const void* opaqueHandle) {
  const auto* allocInfo = static_cast<const AllocationInfo*>(opaqueHandle);

  if (allocInfo->owner != this) {
    // This allocation doesn't belong to this allocator!
    throw E_INVALIDARG;
  }

  return allocInfo->resource.Get();
}

}  // namespace tensorflow

//#endif  // TENSORFLOW_USE_DML
