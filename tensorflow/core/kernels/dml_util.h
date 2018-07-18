#ifndef TENSORFLOW_KERNELS_DML_UTIL_H_
#define TENSORFLOW_KERNELS_DML_UTIL_H_

#include <wrl/client.h>

#include <dml.h>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class DmlUtil {
 public:
  static DML_TENSOR_DESC CreateDmlTensorDesc(const Tensor* tensor);

  static DML_TENSOR_DESC CreateDmlTensorDesc(const Tensor* tensor,
                                             const Tensor* other_tensor);

  static void ConvertNhwcToNchwUsingStrides(DML_TENSOR_DESC& dml_tensor_desc);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DML_UTIL_H_