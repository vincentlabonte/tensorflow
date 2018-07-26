/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_OPS_DML_COMMON_H_
#define TENSORFLOW_KERNELS_OPS_DML_COMMON_H_

#include <wrl/client.h>

#include <dml.h>

#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/dml_util.h"

namespace tensorflow {

class DmlBinaryOp : public BinaryOpShared {
 public:
  explicit DmlBinaryOp(OpKernelConstruction* ctx)
      : BinaryOpShared(ctx, DataTypeToEnum<float>::v(),
                       DataTypeToEnum<float>::v()) {}

  void Compute(OpKernelContext* ctx) override;

  virtual DML_ELEMENT_WISE_FUNCTION GetDmlElementWiseFunction() = 0;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_OPS_DML_COMMON_H_
