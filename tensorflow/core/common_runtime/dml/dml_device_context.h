#ifndef TENSORFLOW_COMMON_RUNTIME_DML_DML_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMMON_RUNTIME_DML_DML_DEVICE_CONTEXT_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_allocator.h"
#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

class DmlDeviceContext : public DeviceContext {
 public:
  DmlDeviceContext() {}

  ~DmlDeviceContext() override {}

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;

 private:
  static void MapAndCopyToResource(ID3D12Resource* resource, const void* src,
                                   UINT64 num_bytes);
  static void MapCopyFromResource(ID3D12Resource* resource, void* dest,
                                  UINT64 num_bytes);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DML_DML_DEVICE_CONTEXT_H_
