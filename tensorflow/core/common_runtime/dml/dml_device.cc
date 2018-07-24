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

#include "tensorflow/core/common_runtime/dml/dml_device.h"

namespace tensorflow {

DmlDevice::DmlDevice(const SessionOptions& options, const string& name,
                     Bytes memory_limit, const DeviceLocality& locality,
                     Allocator* cpu_allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_DML, memory_limit, locality)),
      cpu_allocator_(cpu_allocator),
      device_context_(new DmlDeviceContext()),
      compute_fence_value_(0),
      copy_fence_value_(0),
      is_dml_device_context_open_(false),
      is_copy_command_list_open_(false) {
  THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&d3d12_device_)));
  THROW_IF_FAILED(
      DMLCreateDevice(d3d12_device_.Get(), IID_PPV_ARGS(&dml_device_)));

  D3D12_COMMAND_QUEUE_DESC compute_command_queue_desc = {
      D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
  THROW_IF_FAILED(d3d12_device_->CreateCommandQueue(
      &compute_command_queue_desc, IID_PPV_ARGS(&compute_command_queue_)));

  D3D12_COMMAND_QUEUE_DESC copy_command_queue_desc = {
      D3D12_COMMAND_LIST_TYPE_COPY, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
  THROW_IF_FAILED(d3d12_device_->CreateCommandQueue(
      &copy_command_queue_desc, IID_PPV_ARGS(&copy_command_queue_)));

  THROW_IF_FAILED(d3d12_device_->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&copy_command_allocator_)));

  THROW_IF_FAILED(d3d12_device_->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_COPY, copy_command_allocator_.Get(), nullptr,
      IID_PPV_ARGS(&copy_command_list_)));

  dml_allocator_ = new DmlAllocator(
      d3d12_device_.Get(), CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
      D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      D3D12_RESOURCE_STATE_COMMON);

  THROW_IF_FAILED(d3d12_device_.Get()->CreateFence(
      compute_fence_value_, D3D12_FENCE_FLAG_SHARED,
      IID_PPV_ARGS(&compute_fence_)));
  THROW_IF_FAILED(d3d12_device_.Get()->CreateFence(
      copy_fence_value_, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&copy_fence_)));

  THROW_IF_FAILED(dml_device_->CreateDeviceContext(compute_fence_.Get(),
                                                   &dml_device_context_));
}

DmlDevice::~DmlDevice() {
  delete dml_allocator_;
  device_context_->Unref();
}

void DmlDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  assert(context);
  op_kernel->Compute(context);
}

Allocator* DmlDevice::GetAllocator(AllocatorAttributes attr) {
  if (attr.on_host())
    return cpu_allocator_;
  else
    return dml_allocator_;
}

Status DmlDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                      const AllocatorAttributes alloc_attrs,
                                      Tensor* tensor) {
  AllocatorAttributes attr;
  attr.set_on_host(true);
  Allocator* host_alloc = GetAllocator(attr);

  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy.IsInitialized()) {
      return errors::ResourceExhausted(
          "OOM when allocating tensor of shape ", parsed.shape().DebugString(),
          " and type ", DataTypeString(parsed.dtype()));
    }

    device_context_->CopyCPUTensorToDevice(
        &parsed, this, &copy, [&status](const Status& s) { status = s; });
    *tensor = copy;
  }
  return status;
}

Status DmlDevice::FillContextMap(const Graph* graph,
                                 DeviceContextMap* device_context_map) {
  // Fill in the context map.  It is OK for this map to contain
  // duplicate DeviceContexts so long as we increment the refcount.
  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;
  }

  return Status::OK();
}

Status DmlDevice::Sync() {
  AwaitComputeExecution();
  AwaitCopyExecution();
  return Status::OK();
}

ID3D12Device* DmlDevice::GetD3D12Device() const { return d3d12_device_.Get(); }

IDMLDevice* DmlDevice::GetDmlDevice() const { return dml_device_.Get(); }

IDMLDeviceContext* DmlDevice::GetDmlDeviceContext() const {
  return dml_device_context_.Get();
}

HRESULT DmlDevice::AddComputeOperation(IDMLOperation* operation,
                                       IDMLResource* const* input_resources,
                                       UINT input_count,
                                       IDMLResource* const* output_resources,
                                       UINT output_count) {
  dml_device_context_mu_.lock();
  if (!is_dml_device_context_open_) {
    compute_fence_value_mu_.lock();
    dml_device_context_->Open(compute_fence_value_ + 1);
    compute_fence_value_mu_.unlock();
    is_dml_device_context_open_ = true;
  }
  HRESULT ret = dml_device_context_->AddOperation(
      operation, input_resources, input_count, output_resources, output_count);
  dml_device_context_mu_.unlock();
  return ret;
}

void DmlDevice::AddCopyOperation(ID3D12Resource* dst_resource,
                      ID3D12Resource* src_resource) {
  copy_command_list_mu_.lock();
  if (!is_copy_command_list_open_) {
    THROW_IF_FAILED(d3d12_device_->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_COPY, copy_command_allocator_.Get(), nullptr,
        IID_PPV_ARGS(&copy_command_list_)));
    is_copy_command_list_open_ = true;
  }
  copy_command_list_->CopyResource(dst_resource, src_resource);
  copy_command_list_pending_resource_.push_back(dst_resource);
  copy_command_list_pending_resource_.push_back(src_resource);
  copy_command_list_mu_.unlock();
} 

void DmlDevice::AwaitComputeExecution() {
  dml_device_context_mu_.lock();
  if (is_dml_device_context_open_) {
    ComPtr<ID3D12CommandList> compute_command_list;
    THROW_IF_FAILED(dml_device_context_->Close(&compute_command_list));
    ID3D12CommandList* compute_command_lists[1] = {compute_command_list.Get()};
    compute_command_queue_->ExecuteCommandLists(1, compute_command_lists);
    is_dml_device_context_open_ = false;
  }
  dml_device_context_mu_.unlock();

  HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  compute_fence_value_mu_.lock();
  ++compute_fence_value_;
  THROW_IF_FAILED(compute_command_queue_->Signal(compute_fence_.Get(),
                                                 compute_fence_value_));
  THROW_IF_FAILED(
      compute_fence_->SetEventOnCompletion(compute_fence_value_, fence_event));
  compute_fence_value_mu_.unlock();

  DWORD retVal = WaitForSingleObject(fence_event, INFINITE);
  if (retVal != WAIT_OBJECT_0) {
    DebugBreak();
  }
}

void DmlDevice::AwaitCopyExecution() {
  copy_command_list_mu_.lock();
  ComPtr<ID3D12GraphicsCommandList> temp_copy_command_list = copy_command_list_;
  std::vector<ComPtr<ID3D12Resource>> temp_copy_command_list_pending_resource =
      copy_command_list_pending_resource_;
  if (is_copy_command_list_open_) {
    temp_copy_command_list->Close();
    ID3D12CommandList* copy_command_lists[1] = {temp_copy_command_list.Get()};
    copy_command_queue_->ExecuteCommandLists(1, copy_command_lists);
	is_copy_command_list_open_ = false;
    copy_command_list_ = nullptr;
    copy_command_list_pending_resource_.clear();
  }
  copy_command_list_mu_.unlock(); 

  HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  copy_fence_value_mu_.lock();
  ++copy_fence_value_;
  THROW_IF_FAILED(
      copy_command_queue_->Signal(copy_fence_.Get(), copy_fence_value_));
  THROW_IF_FAILED(
      copy_fence_->SetEventOnCompletion(copy_fence_value_, fence_event));
  copy_fence_value_mu_.unlock();

  DWORD retVal = WaitForSingleObject(fence_event, INFINITE);
  if (retVal != WAIT_OBJECT_0) {
    DebugBreak();
  }

  temp_copy_command_list = nullptr;
  temp_copy_command_list_pending_resource.clear();
}

}  // namespace tensorflow

//#endif  // TENSORFLOW_USE_DML
