#include "tensorflow/core/kernels/dml_activation_ops.h"

namespace tensorflow {

void DmlActivationOp::Compute(OpKernelContext* ctx) {
  DmlOpKernel::Compute(ctx);

  const Tensor& input = ctx->input(0);
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {0}, 0, input.shape(), &output));

  if (!ctx->status().ok()) return;
  if (input.NumElements() <= 0) return;
  if (input.dims() > DML_TENSOR_DIMENSION_COUNT_NCHW) return;

  const void* input_data = input.tensor_data().data();
  const void* output_data = output->tensor_data().data();

  ComPtr<ID3D12Resource> input_resource =
      allocator_->DecodeDataHandle(input_data);
  ComPtr<ID3D12Resource> output_resource =
      allocator_->DecodeDataHandle(output_data);

  ComPtr<IDMLResource> input_dml_resource;
  ComPtr<IDMLResource> output_dml_resource;

  THROW_IF_FAILED(dml_device_context_->CreateResource(input_resource.Get(),
                                                      &input_dml_resource));
  THROW_IF_FAILED(dml_device_context_->CreateResource(output_resource.Get(),
                                                      &output_dml_resource));

  const DML_TENSOR_DESC dml_input_desc = CreateDmlTensorDesc(&input);
  const DML_TENSOR_DESC dml_output_desc = CreateDmlTensorDesc(output);

  ComPtr<IDMLOperation> dml_operation;
  THROW_IF_FAILED(dml_device_->CreateActivationOperation(
      GetDmlActivationFunction(), &dml_input_desc, nullptr, &dml_output_desc,
      nullptr, DML_EXECUTION_HINT_FLAGS_NONE, &dml_operation));

  IDMLResource* input_resources[1] = {input_dml_resource.Get()};
  THROW_IF_FAILED(
      device_->AddComputeOperation(dml_operation.Get(), input_resources, 1,
                                   output_dml_resource.GetAddressOf(), 1));
}

class DmlReluOp : public DmlActivationOp {
 public:
  explicit DmlReluOp(OpKernelConstruction* ctx) : DmlActivationOp(ctx) {}

  DML_ACTIVATION_FUNCTION GetDmlActivationFunction() override {
    return DML_ACTIVATION_FUNCTION_RELU;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device(DEVICE_DML).TypeConstraint<float>("T"), DmlReluOp);

class DmlSoftmaxOp : public DmlActivationOp {
 public:
  explicit DmlSoftmaxOp(OpKernelConstruction* ctx) : DmlActivationOp(ctx){};

  virtual DML_ACTIVATION_FUNCTION GetDmlActivationFunction() {
    return DML_ACTIVATION_FUNCTION_SOFTMAX;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_DML).TypeConstraint<float>("T"),
    DmlSoftmaxOp);

}  // namespace tensorflow
