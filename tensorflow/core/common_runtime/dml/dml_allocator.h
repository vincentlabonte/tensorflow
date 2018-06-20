///* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================*/
//
////#if !TENSORFLOW_USE_DML
////#error This file must only be included when building TensorFlow with DML support
////#endif
//
//#ifndef TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_
//#define TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_
//
//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/framework/allocator.h"
//#include "tensorflow/core/platform/mutex.h"
//#include "tensorflow/core/platform/types.h"
//
//namespace tensorflow {
//
//class DmlAllocator : public Allocator {
// public:
//  DmlAllocator();
//  virtual ~DmlAllocator() override;
//  string Name() override;
//  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
//  void DeallocateRaw(void* ptr) override;
//
//  virtual bool ShouldAllocateEmptyTensors() override final { return true; }
//
// private:
//  mutable mutex mu_;
//  Eigen::SyclDevice* sycl_device_ GUARDED_BY(mu_);  // owned
//  AllocatorStats stats_ GUARDED_BY(mu_);
//
//  TF_DISALLOW_COPY_AND_ASSIGN(DmlAllocator);
//};
//
//}  // namespace tensorflow
//
//#endif  // TENSORFLOW_COMMON_RUNTIME_DML_DML_ALLOCATOR_H_
