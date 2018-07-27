#ifndef TENSORFLOW_KERNELS_DML_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_DML_OPS_COMMON_H_

#include <wrl/client.h>

#include <dml.h>

#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/kernels/dml_util.h"

namespace tensorflow {

class DmlOpKernel : public OpKernel {
 public:
  explicit DmlOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override;

 protected:
  DmlDevice* device_;
  DmlAllocator* allocator_;
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<IDMLDeviceContext> dml_device_context_;

  static DML_TENSOR_DESC CreateDmlTensorDesc(const Tensor* tensor);
  static DML_TENSOR_DESC CreateDmlTensorDesc(const Tensor* tensor,
                                             const Tensor* other_tensor);

  static void ConvertNhwcToNchwUsingStrides(DML_TENSOR_DESC& dml_tensor_desc);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DML_OPS_COMMON_H_
