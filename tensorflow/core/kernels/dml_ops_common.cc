#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

void DmlOpKernel::Compute(OpKernelContext* ctx) {
  DeviceBase* device = ctx->device();
  device_ = dynamic_cast<DmlDevice*>(device);
  OP_REQUIRES(
      ctx, device_,
      errors::Internal("Device should be DML, but is: ", device->name()));

  AllocatorAttributes attrs;
  Allocator* allocator = device_->GetAllocator(attrs);
  allocator_ = dynamic_cast<DmlAllocator*>(allocator);
  OP_REQUIRES(
      ctx, allocator_,
      errors::Internal("Allocator should be DML, but is: ", allocator->Name()));

  dml_device_ = device_->GetDmlDevice();
  dml_device_context_ = device_->GetDmlDeviceContext();
  device_->AwaitCopyExecution();
}

/* static */ DML_TENSOR_DESC DmlOpKernel::CreateDmlTensorDesc(
    const Tensor* tensor) {
  if (tensor->dtype() != DataType::DT_FLOAT) throw E_INVALIDARG;
  int dims = tensor->dims();
  if (dims > DML_TENSOR_DIMENSION_COUNT_NCHW) throw E_INVALIDARG;
  DML_TENSOR_DESC dml_tensor_desc = {DML_TENSOR_DATA_TYPE_FLOAT32,
                                     DML_TENSOR_FLAGS_NONE,
                                     DML_TENSOR_DIMENSION_COUNT_NCHW,
                                     {1, 1, 1, 1}};
  auto dim_sizes = tensor->shape().dim_sizes();
  for (int i = 0; i < dims; i++) {
    dml_tensor_desc.sizes[DML_TENSOR_DIMENSION_COUNT_NCHW - 1 - i] =
        dim_sizes[dims - 1 - i];
  }
  return dml_tensor_desc;
}

/* static */ DML_TENSOR_DESC DmlOpKernel::CreateDmlTensorDesc(
    const Tensor* tensor, const Tensor* other_tensor) {
  if (tensor->dtype() != DataType::DT_FLOAT) throw E_INVALIDARG;
  int dims = tensor->dims();
  int other_dims = other_tensor->dims();
  int max_dims = std::max(dims, other_dims);
  if (dims > DML_TENSOR_DIMENSION_COUNT_NCHW) throw E_INVALIDARG;
  DML_TENSOR_DESC dml_tensor_desc = {DML_TENSOR_DATA_TYPE_FLOAT32,
                                     DML_TENSOR_FLAGS_USE_STRIDES,
                                     DML_TENSOR_DIMENSION_COUNT_NCHW,
                                     {1, 1, 1, 1}};
  auto dim_sizes = tensor->shape().dim_sizes();
  auto other_dim_sizes = other_tensor->shape().dim_sizes();
  UINT stride_value = 1u;
  for (int i = max_dims - 1; i >= 0; i--) {
    if (i >= max_dims - dims && i >= max_dims - other_dims) {
      int64 max_dim_size = std::max(dim_sizes[i], other_dim_sizes[i]);
      if (dim_sizes[i] == 1) {
        dml_tensor_desc.strides[i] = 0;
      } else if (dim_sizes[i] == max_dim_size) {
        dml_tensor_desc.strides[i] = stride_value;
      } else {
        throw E_INVALIDARG;
      }
      dml_tensor_desc.sizes[i] = max_dim_size;
      stride_value *= max_dim_size;
    } else if (i >= max_dims - other_dims) {
      dml_tensor_desc.strides[i] = 0;
      dml_tensor_desc.sizes[i] = other_dim_sizes[i];
      stride_value *= other_dim_sizes[i];
    } else if (i >= max_dims - dims) {
      dml_tensor_desc.strides[i] = stride_value;
      dml_tensor_desc.sizes[i] = dim_sizes[i];
      stride_value *= dim_sizes[i];
    } else {
      throw E_INVALIDARG;
    }
  }
  return dml_tensor_desc;
}

/* static */ void DmlOpKernel::ConvertNhwcToNchwUsingStrides(
    DML_TENSOR_DESC& dml_tensor_desc) {
  UINT val_stride_dml_tensor_desc = 1;
  for (int i = DML_TENSOR_DIMENSION_COUNT_NCHW - 1; i >= 0; i--) {
    if (dml_tensor_desc.sizes[i] > 1) {
      dml_tensor_desc.strides[i] = val_stride_dml_tensor_desc;
    }
    val_stride_dml_tensor_desc *= dml_tensor_desc.sizes[i];
  }

  const UINT dml_tensor_desc_sizes[5] = {
      dml_tensor_desc.sizes[0], dml_tensor_desc.sizes[3],
      dml_tensor_desc.sizes[1], dml_tensor_desc.sizes[2]};
  const UINT dml_tensor_desc_strides[5] = {
      dml_tensor_desc.strides[0], dml_tensor_desc.strides[3],
      dml_tensor_desc.strides[1], dml_tensor_desc.strides[2]};

  for (int i = DML_TENSOR_DIMENSION_COUNT_NCHW - 1; i >= 0; i--) {
    dml_tensor_desc.sizes[i] = dml_tensor_desc_sizes[i];
    dml_tensor_desc.strides[i] = dml_tensor_desc_strides[i];
  }

  dml_tensor_desc.flags = DML_TENSOR_FLAGS_USE_STRIDES;
}

}  // namespace tensorflow
