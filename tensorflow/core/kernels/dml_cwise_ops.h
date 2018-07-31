#ifndef TENSORFLOW_KERNELS_DML_CWISE_OPS_H_
#define TENSORFLOW_KERNELS_DML_CWISE_OPS_H_

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

class DmlBinaryOp : public DmlOpKernel {
 public:
  explicit DmlBinaryOp(OpKernelConstruction* ctx) : DmlOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;

  virtual DML_ELEMENT_WISE_FUNCTION GetDmlElementWiseFunction() = 0;
};

class DmlAddBinaryOp : public DmlBinaryOp {
 public:
  explicit DmlAddBinaryOp(OpKernelConstruction* ctx) : DmlBinaryOp(ctx) {}

  DML_ELEMENT_WISE_FUNCTION GetDmlElementWiseFunction() override;
};

class DmlSubBinaryOp : public DmlBinaryOp {
 public:
  explicit DmlSubBinaryOp(OpKernelConstruction* ctx) : DmlBinaryOp(ctx) {}

  DML_ELEMENT_WISE_FUNCTION GetDmlElementWiseFunction() override;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DML_CWISE_OPS_H_
