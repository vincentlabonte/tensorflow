/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

//#if TENSORFLOW_USE_DML

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_device_context.h"

namespace tensorflow {

void DmlDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  *device_tensor = *cpu_tensor;
  done(Status::OK());
}

void DmlDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             StringPiece edge_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  *cpu_tensor = *device_tensor;
  done(Status::OK());
}

}  // namespace tensorflow
//#endif  // TENSORFLOW_USE_DML
