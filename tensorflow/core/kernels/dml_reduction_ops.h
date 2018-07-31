#ifndef TENSORFLOW_KERNELS_DML_REDUCTION_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_DML_REDUCTION_OPS_COMMON_H_

#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

// For operations where the output is a reduction function along some
// dimensions of the input.
class DmlReductionOp : public DmlOpKernel {
 public:
  explicit DmlReductionOp(OpKernelConstruction* ctx) : DmlOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override;

 protected:
  virtual DML_REDUCE_FUNCTION GetDmlReduceFunction() = 0;

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

class DmlMaxOp : public DmlReductionOp {
 public:
  explicit DmlMaxOp(OpKernelConstruction* ctx) : DmlReductionOp(ctx) {}

 protected:
  DML_REDUCE_FUNCTION GetDmlReduceFunction() override {
    return DML_REDUCE_FUNCTION_MAX;
  }
};

class DmlMeanOp : public DmlReductionOp {
 public:
  explicit DmlMeanOp(OpKernelConstruction* ctx) : DmlReductionOp(ctx) {}

 protected:
  DML_REDUCE_FUNCTION GetDmlReduceFunction() override {
    return DML_REDUCE_FUNCTION_AVERAGE;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_REDUCTION_OPS_COMMON_H_
