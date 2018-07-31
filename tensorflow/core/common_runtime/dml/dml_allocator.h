#ifndef TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_

#include <d3d12.h>
#include <dml.h>

#include "tensorflow/core/common_runtime/dml/d3dx12.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

struct CD3DX12_RESOURCE_DESC;

namespace tensorflow {

class DmlAllocator : public Allocator {
 public:
  DmlAllocator(ID3D12Device* device, const D3D12_HEAP_PROPERTIES& heap_props,
               D3D12_HEAP_FLAGS heap_flags, D3D12_RESOURCE_FLAGS resource_flags,
               D3D12_RESOURCE_STATES initial_state);
  virtual ~DmlAllocator() override;
  string Name() override;
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  virtual bool ShouldAllocateEmptyTensors() override final { return true; }

  // Returns a weak pointer to the ID3D12Resource associated with an opaque
  // allocation handle returned by AllocateRaw.
  ID3D12Resource* DecodeDataHandle(const void* opaque_handle);

 private:
  static const uint32_t kMinResourceSizeExponent = 16;  // 2^16 = 64KB

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

  ComPtr<ID3D12Device> device_;
  D3D12_HEAP_PROPERTIES heap_properties_;
  D3D12_HEAP_FLAGS heap_flags_;
  D3D12_RESOURCE_FLAGS resource_flags_;
  D3D12_RESOURCE_STATES initial_state_;

  std::vector<Bucket> pool_;

  TF_DISALLOW_COPY_AND_ASSIGN(DmlAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_
