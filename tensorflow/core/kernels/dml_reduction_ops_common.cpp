
#include "tensorflow/core/kernels/dml_reduction_ops_common.h"

namespace tensorflow {

void DmlReductionOp::Compute(OpKernelContext* ctx) {
  const Tensor& data = ctx->input(0);
  const Tensor& axes = ctx->input(1);
  VLOG(1) << "data shape: " << data.shape().DebugString();
  VLOG(1) << "axes      : " << axes.SummarizeValue(10);

  ReductionHelper helper;
  OP_REQUIRES_OK(ctx, helper.Simplify(data, axes, keep_dims_));
  CHECK_GE(helper.ndims(), 0);

  if (helper.ndims() == 0 ||
      (helper.ndims() == 1 && !helper.reduce_first_axis())) {
    // Special case. Reduces nothing.  It is unclear why this is
    // necessary, but tests fail without it.  Look into why this
    // case occurs.
    Tensor out;
    if (!out.CopyFrom(data, helper.out_shape())) {
      ctx->SetStatus(errors::Internal("Error during reduction copy."));
    }
    ctx->set_output(0, out);
    return;
  }

  // We must allocate temp tensors using the same alloc attr as
  // output(0) because it is returned as output(0) in the end.
  const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

  // A temporary tensor whose size matches the size of the reduced
  // output.
  Tensor tmp_out;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                              helper.out_reshape(), &tmp_out, alloc_attr));

  ComPtr<IDXGraphicsAnalysis> ga;
  HRESULT hr = DXGIGetDebugInterface1(0, IID_PPV_ARGS(&ga));
  if (SUCCEEDED(hr)) {
    ga->BeginCapture();
  }

  AllocatorAttributes attrs;
  DmlAllocator* allocator =
      static_cast<DmlAllocator*>(ctx->device()->GetAllocator(attrs));

  const void* input_data = data.tensor_data().data();
  const void* output_data = tmp_out.tensor_data().data();

  ComPtr<ID3D12Resource> input_resource =
      allocator->DecodeDataHandle(input_data);
  ComPtr<ID3D12Resource> output_resource =
      allocator->DecodeDataHandle(output_data);

  DmlInterface* dml_interface = DmlInterface::instance();
  ComPtr<IDMLDevice> dml_device = dml_interface->GetDmlDevice();
  ComPtr<IDMLDeviceContext> dml_device_context;
  THROW_IF_FAILED(dml_device->CreateDeviceContext(
      dml_interface->GetD3D12Fence(), &dml_device_context));

  ComPtr<IDMLResource> input_dml_resource;
  ComPtr<IDMLResource> output_dml_resource;

  THROW_IF_FAILED(dml_device_context->CreateResource(input_resource.Get(),
                                                     &input_dml_resource));
  THROW_IF_FAILED(dml_device_context->CreateResource(output_resource.Get(),
                                                     &output_dml_resource));

  DML_TENSOR_DESC dml_input_desc = DmlUtil::CreateDmlTensorDesc(&data);
  DML_TENSOR_DESC dml_output_desc = DmlUtil::CreateDmlTensorDesc(&tmp_out);

  DmlUtil::ConvertNhwcToNchwUsingStrides(dml_input_desc);
  DmlUtil::ConvertNhwcToNchwUsingStrides(dml_output_desc);

  auto axe_vec = axes.vec<int32>();
  UINT reduction_axes[5];
  for (int i = 0; i < axe_vec.size(); i++) {
    reduction_axes[i] = axe_vec(i);
  }

  ComPtr<IDMLOperation> dml_operation;
  THROW_IF_FAILED(dml_device->CreateReduceOperation(
      GetDmlReduceFunction(), &dml_input_desc, &dml_output_desc, reduction_axes,
      axe_vec.size(), DML_EXECUTION_HINT_FLAGS_NONE, &dml_operation));

  THROW_IF_FAILED(dml_device_context->Open(dml_interface->GetFenceValue() + 1));

  THROW_IF_FAILED(dml_device_context->AddOperation(
      dml_operation.Get(), input_dml_resource.GetAddressOf(), 1,
      output_dml_resource.GetAddressOf(), 1));

  ComPtr<ID3D12CommandList> compute_command_list;
  THROW_IF_FAILED(dml_device_context->Close(&compute_command_list));

  ID3D12CommandList* compute_command_lists[1] = {compute_command_list.Get()};

  dml_interface->GetD3D12CommandQueue()->ExecuteCommandLists(
      1, compute_command_lists);

  dml_interface->AwaitExecution();

  if (SUCCEEDED(hr)) {
    ga->EndCapture();
  }

  // Set the real output using the contents of the reduction but the
  // real expected output shape.  The number of elements should
  // match between the two shapes.
  Tensor out;
  if (!out.CopyFrom(tmp_out, helper.out_shape())) {
    ctx->SetStatus(errors::Internal("Error during reduction copy."));
  }
  ctx->set_output(0, out);
}


REGISTER_KERNEL_BUILDER(Name("Max")
                            .Device(DEVICE_DML)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("reduction_indices"),
                        DmlMaxOp);

}  // namespace tensorflow
