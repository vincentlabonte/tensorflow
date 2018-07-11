#ifndef TENSORFLOW_KERNELS_DML_UTIL_H_
#define TENSORFLOW_KERNELS_DML_UTIL_H_

#include <wrl/client.h>

#include <DXProgrammableCapture.h>
#include <dml.h>
#include <dxgi1_5.h>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class DmlUtil {
 public:
  static DML_TENSOR_DESC CreateDmlTensorDesc(const Tensor* tensor);

  static DML_TENSOR_DESC CreateDmlTensorDesc(const Tensor* tensor,
                                             const Tensor* other_tensor);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DML_UTIL_H_