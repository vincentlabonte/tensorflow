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

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dml/dml_device.h"

#include "tensorflow/core/common_runtime/dml/dml_util.h"

namespace tensorflow {

class DmlDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    auto dmlInterface = DmlInterface::instance();

    size_t n = 1;
    auto iter = options.config.device_count().find("DML");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }

    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:DML:", i);
      devices->push_back(new DmlDevice(
          options, name, Bytes(256 << 20), DeviceLocality(), "",
          dmlInterface->GetDmlAllocator(), dmlInterface->GetCPUAllocator(),
          dmlInterface->GetDmlContext()));
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("DML", DmlDeviceFactory, 200);

}  // namespace tensorflow

//#endif  // TENSORFLOW_USE_DML
