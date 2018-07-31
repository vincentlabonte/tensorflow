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

  HRESULT AddComputeOperation(IDMLOperation* operation,
                              IDMLResource* const* input_resources,
                              UINT input_count,
                              IDMLResource* const* output_resources,
                              UINT output_count);

  void AddCopyOperation(ID3D12Resource* dst_resource,
                        ID3D12Resource* src_resource);

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
  mutex copy_command_list_mu_;
  ComPtr<ID3D12GraphicsCommandList> copy_command_list_
      GUARDED_BY(copy_command_list_mu_);
  bool is_copy_command_list_open_ GUARDED_BY(copy_command_list_mu_);
  std::vector<ComPtr<ID3D12Resource>> copy_command_list_pending_resource_
      GUARDED_BY(copy_command_list_mu_);
  ComPtr<ID3D12Fence> copy_fence_;
  mutex copy_fence_value_mu_;
  uint64_t copy_fence_value_ GUARDED_BY(copy_fence_value_mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_DEVICE_H_
