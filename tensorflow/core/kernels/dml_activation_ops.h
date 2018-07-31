#ifndef TENSORFLOW_KERNELS_DML_ACTIVATION_OPS_H_
#define TENSORFLOW_KERNELS_DML_ACTIVATION_OPS_H_

#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

class DmlActivationOp : public DmlOpKernel {
 public:
  explicit DmlActivationOp(OpKernelConstruction* ctx) : DmlOpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override;

  virtual DML_ACTIVATION_FUNCTION GetDmlActivationFunction() = 0;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DML_ACTIVATION_OPS_H_
