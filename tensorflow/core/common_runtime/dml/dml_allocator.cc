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
//#ifdef TENSORFLOW_USE_DML
//
//#include "tensorflow/core/common_runtime/dml/dml_allocator.h"
//
//namespace tensorflow {
//
//DmlAllocator::DmlAllocator(Eigen::QueueInterface* queue)
//    : sycl_device_(new Eigen::SyclDevice(queue)) {
//  cl::sycl::queue& sycl_queue = sycl_device_->sycl_queue();
//  const cl::sycl::device& device = sycl_queue.get_device();
//  stats_.bytes_limit =
//      device.get_info<cl::sycl::info::device::max_mem_alloc_size>();
//}
//
//DmlAllocator::~DmlAllocator() {
//  if (dml_device_) {
//    delete dml_device_;
//  }
//}
//
//string DmlAllocator::Name() { return "device:DML"; }
//
//void* DmlAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
//  return nullptr;
//}
//
//void DmlAllocator::DeallocateRaw(void* ptr) {
//}
//
//void DmlAllocator::GetStats(AllocatorStats* stats) {
//}
//
//void DmlAllocator::ClearStats() override {
//}
//
//size_t DmlAllocator::RequestedSize(void* ptr) {
//  return 0;
//}
//
//}  // namespace tensorflow
//
//#endif  // TENSORFLOW_USE_DML
