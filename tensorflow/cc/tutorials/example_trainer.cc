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
#include <wrl/client.h>

#include <d3d12sdklayers.h>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
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

#include <DXGItype.h>  
#include <dxgi1_2.h>  
#include <dxgi1_3.h>  
#include <DXProgrammableCapture.h>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

using namespace tensorflow;

int main(int argc, char* argv[]) {
  std::ifstream file(
      "C:\\Users\\t-vilab.REDMOND\\Desktop\\onnx_to_tf\\squeezenet.pb",
      std::ios::in | std::ios::binary);
  GraphDef def;
  if (!def.ParseFromIstream(&file)) {
    return -1;
  }
  file.close();

  // Creates a session.
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  if (options.target.empty()) {
    // graph::SetDefaultDevice("/cpu:0", &def);
    // graph::SetDefaultDevice("/device:GPU:0", &def);
    graph::SetDefaultDevice("/device:DML:0", &def);
  }
  TF_CHECK_OK(session->Create(def));
  // NumElements = 150528
  Tensor tensor(DataType::DT_FLOAT, {1, 3, 224, 224});
  std::ifstream img(
      "C:\\Users\\t-vilab.REDMOND\\Desktop\\onnx_to_tf\\img\\pug3.txt",
      std::ios::in | std::ios::binary);
  std::vector<float> vec;
  float f;
  while (img.read(reinterpret_cast<char*>(&f), sizeof(f))) vec.push_back(f);
  img.close();
  std::copy_n(vec.begin(), vec.size(), tensor.flat<float>().data());

  IDXGraphicsAnalysis* ga;
  HRESULT hr =
      DXGIGetDebugInterface1(0, __uuidof(ga), reinterpret_cast<void**>(&ga));
  if (SUCCEEDED(hr)) {
    ga->BeginCapture();
  }

  auto t1 = Clock::now();

  std::vector<Tensor> outputs;
  TF_CHECK_OK(
      session->Run({{"data_0:0", tensor}}, {"Reshape_1:0"}, {}, &outputs));
  CHECK_EQ(size_t{1}, outputs.size());
  const Tensor& tensor_result = outputs[0];

  auto t2 = Clock::now();
  std::cout
      << "Delta t2-t1: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
      << " nanoseconds" << std::endl;

  if (SUCCEEDED(hr)) {
    ga->EndCapture();
  }

  int max_index = 0;
  float max_value = tensor_result.flat<float>()(0);
  for (int i = 1; i < 1000; ++i) {
    float value = tensor_result.flat<float>()(i);
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }
  printf("%i\n%f\n", max_index, max_value);
}
