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
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using namespace tensorflow;

int main(int argc, char* argv[]) {
  const int dataCount = 5;
  std::vector<float> data_a({1, 6, -4, 8, -7});
  std::vector<float> data_b({-3, 2, 4, -5, -9});

  Scope root = Scope::NewRootScope();

  auto a = ops::Placeholder(root.WithOpName("a"), DataType::DT_FLOAT,
                            ops::Placeholder::Shape({dataCount}));
  auto b = ops::Placeholder(root.WithOpName("b"), DataType::DT_FLOAT,
                            ops::Placeholder::Shape({dataCount}));
  //auto b = ops::Const(root.WithOpName("b"), {-3, 2, 4, -5, -9}, {dataCount});
  auto result = ops::Add(root.WithOpName("result"), a, b);

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

  // Creates a session.
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  if (options.target.empty()) {
    // graph::SetDefaultDevice("/cpu:0", &def);
    // graph::SetDefaultDevice("/device:GPU:0", &def);
    graph::SetDefaultDevice("/device:DML:0", &def);
  }
  TF_CHECK_OK(session->Create(def));

  Tensor tensor_a(DataType::DT_FLOAT, {dataCount});
  std::copy_n(data_a.begin(), data_a.size(), tensor_a.flat<float>().data());
  Tensor tensor_b(DataType::DT_FLOAT, {dataCount});
  std::copy_n(data_b.begin(), data_b.size(), tensor_b.flat<float>().data());

  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({{"a:0", tensor_a}, {"b:0", tensor_b}}, {"result:0"},
                           {}, &outputs));
  CHECK_EQ(size_t{1}, outputs.size());
  const Tensor& tensor_result = outputs[0];

  for (int i = 0; i < dataCount; ++i) {
    printf("%f + %f = %f\n", data_a[i], data_b[i],
           tensor_result.vec<float>()(i));
  }
}
