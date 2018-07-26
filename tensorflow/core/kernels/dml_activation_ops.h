#ifndef TENSORFLOW_KERNELS_DML_ACTIVATION_OPS_H_
#define TENSORFLOW_KERNELS_DML_ACTIVATION_OPS_H_

#include <wrl/client.h>

#include <dml.h>

#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/kernels/dml_util.h"

namespace tensorflow {

class DmlActivationOp : public OpKernel {
 public:
  explicit DmlActivationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override;

  virtual DML_ACTIVATION_FUNCTION GetDmlActivationFunction() = 0;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DML_ACTIVATION_OPS_H_