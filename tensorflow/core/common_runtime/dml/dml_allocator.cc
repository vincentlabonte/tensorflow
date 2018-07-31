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
  size_t requested_size = 0;
};

DmlAllocator::DmlAllocator(ID3D12Device* device,
                           const D3D12_HEAP_PROPERTIES& heap_props,
                           D3D12_HEAP_FLAGS heap_flags,
                           D3D12_RESOURCE_FLAGS resource_flags,
                           D3D12_RESOURCE_STATES initial_state)
    : device_(device),
      heap_properties_(heap_props),
      heap_flags_(heap_flags),
      resource_flags_(resource_flags),
      initial_state_(initial_state) {}

DmlAllocator::~DmlAllocator() {}

/*static*/ std::ptrdiff_t DmlAllocator::GetBucketIndexFromSize(uint64_t size) {
  assert(size != 0);

  // Each bucket is twice as large as the previous one, in ascending order
  std::ptrdiff_t index = static_cast<std::ptrdiff_t>(ceil(log2(size)));
  assert((1ull << index) >= size);  // This must be true unless there were some
                                    // strange rounding issues

  // The smallest bucket is 2^n bytes large, where n = kMinResourceSizeExponent
  index = std::max<std::ptrdiff_t>(index, kMinResourceSizeExponent);
  index -= kMinResourceSizeExponent;

  return index;
}

/*static*/ uint64_t DmlAllocator::GetBucketSizeFromIndex(std::ptrdiff_t index) {
  return (1ull << (index + kMinResourceSizeExponent));
}

string DmlAllocator::Name() { return "device:DML"; }

void* DmlAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  size_t size = std::max<size_t>(1, num_bytes);

  // Find the bucket for this allocation size
  std::ptrdiff_t bucket_index = GetBucketIndexFromSize(size);

  if (pool_.size() <= bucket_index) {
    // Ensure there are sufficient buckets
    pool_.resize(bucket_index + 1);
  }

  Bucket& bucket = pool_[bucket_index];
  const uint64_t bucket_size = GetBucketSizeFromIndex(bucket_index);

  ComPtr<ID3D12Resource> resource;
  bucket.mu.lock();
  if (bucket.resources.empty()) {
    bucket.mu.unlock();
    // No more resources in this bucket - allocate a new one
    THROW_IF_FAILED(device_->CreateCommittedResource(
        &heap_properties_, heap_flags_,
        &CD3DX12_RESOURCE_DESC::Buffer(bucket_size, resource_flags_),
        initial_state_, nullptr, IID_PPV_ARGS(&resource)));
  } else {
    // Retrieve a resource from the bucket
    resource = std::move(bucket.resources.back());
    bucket.resources.pop_back();
    bucket.mu.unlock();
  }

  assert(resource != nullptr);
  assert(resource->GetDesc().Width == bucket_size);

  auto alloc_info = std::make_unique<AllocationInfo>();
  alloc_info->owner = this;
  alloc_info->resource =
      std::move(resource);  // Take ownership of this resource
  alloc_info->requested_size = size;

  return alloc_info.release();  // "Detach"
}

void DmlAllocator::DeallocateRaw(void* ptr) {
  std::unique_ptr<AllocationInfo> alloc_info(static_cast<AllocationInfo*>(ptr));

  assert(alloc_info != nullptr);  // Can't free nullptr

  if (alloc_info->owner != this) {
    // This allocation doesn't belong to this allocator!
    alloc_info.release();
    throw E_INVALIDARG;
  }

  std::ptrdiff_t bucket_index =
      GetBucketIndexFromSize(alloc_info->requested_size);

  // The resource size must match the bucket size...
  assert(GetBucketSizeFromIndex(bucket_index) ==
         alloc_info->resource->GetDesc().Width);
  assert(pool_.size() > bucket_index);

  // Return the resource to the bucket
  Bucket& bucket = pool_[bucket_index];
  bucket.mu.lock();
  bucket.resources.push_back(std::move(alloc_info->resource));
  bucket.mu.unlock();

  // Free the allocation info
  alloc_info.reset();
}

ID3D12Resource* DmlAllocator::DecodeDataHandle(const void* opaque_handle) {
  const auto* alloc_info = static_cast<const AllocationInfo*>(opaque_handle);

  if (alloc_info->owner != this) {
    // This allocation doesn't belong to this allocator!
    throw E_INVALIDARG;
  }

  return alloc_info->resource.Get();
}

}  // namespace tensorflow
