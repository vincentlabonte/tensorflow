#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dml/dml_device.h"

namespace tensorflow {

class DmlDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    size_t n = 1;
    auto iter = options.config.device_count().find("DML");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }

    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:DML:", i);
      devices->push_back(new DmlDevice(
          options, name, Bytes(256 << 20), DeviceLocality(),
          cpu_allocator()));
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("DML", DmlDeviceFactory, 200);

}  // namespace tensorflow
