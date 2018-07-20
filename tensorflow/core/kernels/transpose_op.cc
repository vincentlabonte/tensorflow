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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/transpose_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

#include <wrl/client.h>

#include <dml.h>

#include "tensorflow/core/common_runtime/dml/dml_allocator.h"
#include "tensorflow/core/common_runtime/dml/dml_interface.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/kernels/dml_util.h" 

namespace tensorflow {

// inv = InvertPermutationOp(T<int32/int64> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

template <typename T>
class InvertPermutationOp : public OpKernel {
 public:
  explicit InvertPermutationOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument("invert_permutation expects a 1D vector."));
    auto Tin = input.vec<T>();
    OP_REQUIRES(context,
                FastBoundsCheck(Tin.size(), std::numeric_limits<int32>::max()),
                errors::InvalidArgument("permutation of nonnegative int32s "
                                        "must have <= int32 max elements"));
    const T N = static_cast<T>(Tin.size());  // Safe: bounds-checked above.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto Tout = output->vec<T>();
    std::fill_n(Tout.data(), N, -1);
    for (int i = 0; i < N; ++i) {
      const T d = internal::SubtleMustCopy(Tin(i));
      OP_REQUIRES(context, FastBoundsCheck(d, N),
                  errors::InvalidArgument(d, " is not between 0 and ", N));
      OP_REQUIRES(context, Tout(d) == -1,
                  errors::InvalidArgument(d, " is duplicated in the input."));
      Tout(d) = i;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int64>("T"),
    InvertPermutationOp<int64>);

REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int64>);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

namespace {
template <typename Tperm>
Status PermutationHelper(const Tensor& perm, const int dims,
                         std::vector<int32>* permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return Status::OK();
}
}  // namespace

// output = TransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.

void TransposeOp::Compute(OpKernelContext* ctx) {
  const Tensor& input = ctx->input(0);
  const Tensor& perm = ctx->input(1);
  // Preliminary validation of sizes.
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm.shape()),
              errors::InvalidArgument("perm must be a vector, not ",
                                      perm.shape().DebugString()));

  // Although Tperm may be an int64 type, an int32 is sufficient to hold
  // dimension range values, so the narrowing here should be safe.
  std::vector<int32> permutation;
  const int dims = input.dims();
  if (perm.dtype() == DT_INT32) {
    OP_REQUIRES_OK(ctx, PermutationHelper<int32>(perm, dims, &permutation));
  } else {
    OP_REQUIRES_OK(ctx, PermutationHelper<int64>(perm, dims, &permutation));
  }
  TensorShape shape;

  // Check whether permutation is a permutation of integers of [0 .. dims).
  gtl::InlinedVector<bool, 8> bits(dims);
  bool is_identity = true;
  for (int i = 0; i < dims; ++i) {
    const int32 d = permutation[i];
    OP_REQUIRES(
        ctx, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
    bits[d] = true;
    const auto dim_size = input.dim_size(d);
    shape.AddDim(dim_size);
    if (d != i) {
      is_identity = false;
    }
  }
  for (int i = 0; i < dims; ++i) {
    OP_REQUIRES(
        ctx, bits[i],
        errors::InvalidArgument(i, " is missing from {",
                                str_util::Join(permutation, ","), "}."));
  }

  // 0-D, 1-D, and identity transposes do nothing.
  if (!IsConjugate() && (dims <= 1 || is_identity)) {
    ctx->set_output(0, input);
    return;
  } else if (!IsConjugate() && internal::NonSingletonDimensionsAlign(
                                   input.shape(), permutation)) {
    Tensor output;
    OP_REQUIRES(ctx, output.CopyFrom(input, shape),
                errors::Unknown("Error reshaping Tensor."));
    ctx->set_output(0, output);
    return;
  }

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  if (shape.num_elements() > 0) {
    OP_REQUIRES_OK(ctx, DoTranspose(ctx, input, permutation, output));
  }
}

Status TransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status ConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                            const Tensor& in,
                                            gtl::ArraySlice<int32> perm,
                                            Tensor* out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

#ifdef INTEL_MKL
#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          MklTransposeCpuOp);         \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          MklConjugateTransposeCpuOp);
TF_CALL_ALL_TYPES(REGISTER);
#undef REGISTER

#else  // INTEL_MKL

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeCpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeCpuOp);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER
#endif  // INTEL_MKL

#if GOOGLE_CUDA
Status TransposeGpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<GPUDevice>(), in, perm,
                                   out);
}
Status ConjugateTransposeGpuOp::DoTranspose(OpKernelContext* ctx,
                                            const Tensor& in,
                                            gtl::ArraySlice<int32> perm,
                                            Tensor* out) {
  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<GPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeGpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeGpuOp);
TF_CALL_POD_TYPES(REGISTER);
#undef REGISTER
#endif

#ifdef TENSORFLOW_USE_SYCL
Status TransposeSyclOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                    gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef Eigen::SyclDevice SYCLDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<SYCLDevice>(), in, perm,
                                   out);
}
Status ConjugateTransposeSyclOp::DoTranspose(OpKernelContext* ctx,
                                             const Tensor& in,
                                             gtl::ArraySlice<int32> perm,
                                             Tensor* out) {
  typedef Eigen::SyclDevice SYCLDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<SYCLDevice>(), in,
                                            perm, out);
}
#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_SYCL)    \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeSyclOp);           \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_SYCL)    \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeSyclOp);
TF_CALL_POD_TYPES(REGISTER);
#undef REGISTER
#endif

Status DmlTransposeOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   gtl::ArraySlice<int32> perm, Tensor* out) {
  gtl::InlinedVector<int64, 4> in_dim_sizes = in.shape().dim_sizes();

  if (in.dims() > DML_TENSOR_DIMENSION_COUNT_NCHW) throw E_INVALIDARG;

  // Calculate strides from original shape.
  std::vector<uint32_t> input_strides(in_dim_sizes.size());
  input_strides.back() = 1;
  for (int i = static_cast<int>(input_strides.size()) - 2; i >= 0; i--) {
    input_strides[i] =
        input_strides[i + 1] * static_cast<uint32_t>(in_dim_sizes[i + 1]);
  }

  // Input tensor will use strides to perm the permuted copy.
  DML_TENSOR_DESC dml_input_desc = DmlUtil::CreateDmlTensorDesc(&in);
  dml_input_desc.flags = DML_TENSOR_FLAGS_USE_STRIDES;

  // Output tensor will have same shape as input, but no striding.
  DML_TENSOR_DESC dml_output_desc = DmlUtil::CreateDmlTensorDesc(out);

  // Fill leading tensor desc sizes/strides with defaults.
  const int leadingDims = static_cast<int32_t>(DML_TENSOR_DIMENSION_COUNT_NCHW -
                                               in_dim_sizes.size());
  for (int dimDML = 0; dimDML < leadingDims; ++dimDML) {
    dml_input_desc.sizes[dimDML] = 1;
    dml_output_desc.sizes[dimDML] = 1;
    dml_input_desc.strides[dimDML] = 0;
    dml_output_desc.strides[dimDML] = 0;
  }

  // Permute the shape and strides.
  gtl::InlinedVector<int64, 4> out_dim_sizes;
  for (int dimInput = 0; dimInput < in_dim_sizes.size(); ++dimInput) {
    int dim_dml = dimInput + leadingDims;
    int dimPermuted = perm[dimInput];

    out_dim_sizes[dimInput] = in_dim_sizes[dimPermuted];

    dml_input_desc.sizes[dim_dml] =
        static_cast<int32_t>(in_dim_sizes[dimPermuted]);
    dml_output_desc.sizes[dim_dml] =
        static_cast<int32_t>(in_dim_sizes[dimPermuted]);
    dml_input_desc.strides[dim_dml] = input_strides[dimPermuted];
    dml_output_desc.strides[dim_dml] = 0;
  }

  AllocatorAttributes attrs;
  DmlAllocator* allocator =
      static_cast<DmlAllocator*>(ctx->device()->GetAllocator(attrs));

  const void* input_data = in.tensor_data().data();
  const void* output_data = out->tensor_data().data();

  ComPtr<ID3D12Resource> input_resource =
      allocator->DecodeDataHandle(input_data);
  ComPtr<ID3D12Resource> output_resource =
      allocator->DecodeDataHandle(output_data);

  DmlInterface* dml_interface = DmlInterface::instance();
  ComPtr<IDMLDevice> dml_device = dml_interface->GetDmlDevice();
  ComPtr<IDMLDeviceContext> dml_device_context =
      dml_interface->GetDmlDeviceContext();

  ComPtr<IDMLResource> input_dml_resource;
  ComPtr<IDMLResource> output_dml_resource;

  THROW_IF_FAILED(dml_device_context->CreateResource(input_resource.Get(),
                                                     &input_dml_resource));
  THROW_IF_FAILED(dml_device_context->CreateResource(output_resource.Get(),
                                                     &output_dml_resource));

  const DML_TENSOR_DESC* dml_input_descs[1] = {&dml_input_desc};

  ComPtr<IDMLOperation>
      dml_operation;
  THROW_IF_FAILED(dml_device->CreateElementWiseOperation(
      DML_ELEMENT_WISE_FUNCTION_IDENTITY, dml_input_descs, 1, &dml_output_desc,
      nullptr, DML_EXECUTION_HINT_FLAGS_NONE, &dml_operation));

  IDMLResource* input_resources[1] = {input_dml_resource.Get()};
  THROW_IF_FAILED(dml_interface->AddComputeOperation(
      dml_operation.Get(), input_resources, 1,
                                       output_dml_resource.GetAddressOf(), 1));
}

REGISTER_KERNEL_BUILDER(Name("Transpose")
                            .Device(DEVICE_DML)
                            .TypeConstraint<float>("T")
                            .HostMemory("perm"),
                        DmlTransposeOp);

}  // namespace tensorflow
