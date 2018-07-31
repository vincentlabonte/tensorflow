#include "tensorflow/core/kernels/dml_cwise_ops.h"

namespace tensorflow {

void DmlBinaryOp::Compute(OpKernelContext* ctx) {
  DmlOpKernel::Compute(ctx);

  const Tensor& in0 = ctx->input(0);
  const Tensor& in1 = ctx->input(1);
  BCast bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape()));

  OP_REQUIRES(ctx, bcast.IsValid(),
              errors::InvalidArgument(
                  "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
                  in1.shape().DebugString()));

  Tensor* out = nullptr;
  const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {0, 1}, 0, output_shape, &out));

  if (out->NumElements() == 0) {
    return;
  }

  const void* in0_data = in0.tensor_data().data();
  const void* in1_data = in1.tensor_data().data();
  const void* out_data = out->tensor_data().data();

  ComPtr<ID3D12Resource> in0_resource = allocator_->DecodeDataHandle(in0_data);
  ComPtr<ID3D12Resource> in1_resource = allocator_->DecodeDataHandle(in1_data);
  ComPtr<ID3D12Resource> out_resource = allocator_->DecodeDataHandle(out_data);

  ComPtr<IDMLResource> in0_dml_resource;
  ComPtr<IDMLResource> in1_dml_resource;
  ComPtr<IDMLResource> out_dml_resource;

  THROW_IF_FAILED(dml_device_context_->CreateResource(in0_resource.Get(),
                                                      &in0_dml_resource));
  THROW_IF_FAILED(dml_device_context_->CreateResource(in1_resource.Get(),
                                                      &in1_dml_resource));
  THROW_IF_FAILED(dml_device_context_->CreateResource(out_resource.Get(),
                                                      &out_dml_resource));

  DML_TENSOR_DESC dml_input_desc[2] = {CreateDmlTensorDesc(&in0, &in1),
                                       CreateDmlTensorDesc(&in1, &in0)};

  DML_TENSOR_DESC const* dml_input_ref[2] = {&dml_input_desc[0],
                                             &dml_input_desc[1]};

  DML_TENSOR_DESC dml_output_desc = {CreateDmlTensorDesc(out)};

  ComPtr<IDMLOperation> dml_operation;
  THROW_IF_FAILED(dml_device_->CreateElementWiseOperation(
      GetDmlElementWiseFunction(), &dml_input_ref[0], 2, &dml_output_desc,
      nullptr,  // params
      DML_EXECUTION_HINT_FLAGS_NONE, dml_operation.GetAddressOf()));

  IDMLResource* input_resources[2] = {in0_dml_resource.Get(),
                                      in1_dml_resource.Get()};
  THROW_IF_FAILED(
      device_->AddComputeOperation(dml_operation.Get(), input_resources, 2,
                                   out_dml_resource.GetAddressOf(), 1));
}

DML_ELEMENT_WISE_FUNCTION DmlAddBinaryOp::GetDmlElementWiseFunction() {
  return DML_ELEMENT_WISE_FUNCTION_ADD;
}

REGISTER_KERNEL_BUILDER(
    Name("Add").Device(DEVICE_DML).TypeConstraint<float>("T"), DmlAddBinaryOp);


DML_ELEMENT_WISE_FUNCTION DmlSubBinaryOp::GetDmlElementWiseFunction() {
  return DML_ELEMENT_WISE_FUNCTION_SUBTRACT;
}

REGISTER_KERNEL_BUILDER(
    Name("Sub").Device(DEVICE_DML).TypeConstraint<float>("T"), DmlSubBinaryOp);
REGISTER_KERNEL_BUILDER(Name("Sub")
                            .Device(DEVICE_DML)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::sub<int32>>);

}  // namespace tensorflow
