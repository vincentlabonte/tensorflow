#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_UTIL_H_

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

inline void THROW_IF_FAILED(HRESULT hr) {
  if (FAILED(hr)) {
    throw hr;
  }
}

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DML_DML_UTIL_H_
