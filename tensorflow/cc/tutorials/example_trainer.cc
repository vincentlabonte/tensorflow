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

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;

int main(int argc, char* argv[]) {
  const int32_t dataCount = 5;

  Scope root = Scope::NewRootScope();

  auto a = ops::Const(root.WithOpName("a"), {1, 6, -4, 8, -7}, {dataCount});
  auto b = ops::Const(root.WithOpName("b"), {-3, 2, 4, -5, -9}, {dataCount});

  // x = a * 2
  auto x = ops::Multiply(root.WithOpName("x"), a, 2);

  // y = x + b
  auto y = ops::Add(root.WithOpName("result"), x, b);

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

  // Creates a session.
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  if (options.target.empty()) {
    graph::SetDefaultDevice("/cpu:0", &def);
    // graph::SetDefaultDevice("/device:GPU:0", &def);
    // graph::SetDefaultDevice("/device:DML:0", &def);
  }
  TF_CHECK_OK(session->Create(def));

  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({}, {"result:0", "a:0", "b:0"}, {}, &outputs));
  CHECK_EQ(size_t{3}, outputs.size());

  const Tensor& result = outputs[0];
  const Tensor& as = outputs[1];
  const Tensor& bs = outputs[2];

  for (uint32_t i = 0; i < dataCount; ++i) {
    printf("%i * 2 + %i = %i\n", as.vec<int32_t>()(i), bs.vec<int32_t>()(i),
           result.vec<int32_t>()(i));
  }
}