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
typedef std::chrono::nanoseconds Nanoseconds;

using namespace tensorflow;

std::string DeviceTypeToDevice(std::string device_type);

int main(int argc, char* argv[]) {
  // Read model
  if (argc <= 1) {
    printf("A model path must be specified");
    return -1;
  }

  std::ifstream model_file(argv[1], std::ios::in | std::ios::binary);
  GraphDef def;
  if (!def.ParseFromIstream(&model_file)) {
    printf("Unable to parse model: %s", argv[1]);
    return -1;
  }
  model_file.close();

  // Read image
  if (argc <= 2) {
    printf("An image path must be specified");
    return -1;
  }

  Tensor image_tensor(DataType::DT_FLOAT, {1, 3, 224, 224});
  std::ifstream imgage_file(argv[2], std::ios::in | std::ios::binary);
  std::vector<float> image_data;
  float f;
  while (imgage_file.read(reinterpret_cast<char*>(&f), sizeof(f)))
    image_data.push_back(f);
  imgage_file.close();

  if (image_data.size() != image_tensor.NumElements()) {
    printf(
        "The image must be a binary file and must have shape {1, 3, 224, 224}");
    return -1;
  }
  std::copy_n(image_data.begin(), image_data.size(),
              image_tensor.flat<float>().data());

  // Create session
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));

  std::string device_type = argc <= 3 ? "CPU" : argv [3];
  std::string device = DeviceTypeToDevice(device_type);
  if (device == "") {
    printf("The device type must be \"CPU\" or \"CUDA\" or \"DML\"");
    return -1;
  }

  if (options.target.empty()) {
    graph::SetDefaultDevice(device, &def);
  }

  auto create_status = session->Create(def);
  if (!create_status.ok()) {
    printf("The session creation failed: %s", create_status.error_message().c_str());
    return -1;
  }

  std::vector<Tensor> outputs;
  auto time_before = Clock::now();
  auto run_status =
      session->Run({{"data_0:0", image_tensor}}, {"Reshape_1:0"}, {}, &outputs);
  auto time_after = Clock::now();
  if (!run_status.ok()) {
    printf("The session run failed: %s", run_status.error_message().c_str());
    return -1;
  }

  int64 nanoseconds =
      std::chrono::duration_cast<Nanoseconds>(time_after - time_before).count();
  float time = nanoseconds * 0.000000001;

  const Tensor& tensor_result = outputs[0];
  int max_index = 0;
  float max_value = tensor_result.flat<float>()(0);
  for (int i = 1; i < 1000; ++i) {
    float value = tensor_result.flat<float>()(i);
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  printf(
      "{"
      "\"max_index\":%i,"
      "\"max_value\":%f,"
      "\"device_type\":\"%s\","
      "\"time\":%f"
      "}",
      max_index, max_value, device_type.c_str(), time);
}

std::string DeviceTypeToDevice(std::string device_type) {
  if (device_type == "CPU") {
    return "/cpu:0";
  } else if (device_type == "CUDA") {
    return "/device:GPU:0";
  } else if (device_type == "DML") {
    return "/device:DML:0";
  } else {
    return "";
  }
}
