#ifndef K2_CSRC_DEVICE_GUARD_H_
#define K2_CSRC_DEVICE_GUARD_H_

#include "k2/csrc/context.h"

class DeviceGuard {
 public:
  explicit DeviceGuard(k2::ContextPtr c) {}
  explicit DeviceGuard(int32_t index) {}
};

#endif // K2_CSRC_DEVICE_GUARD_H_
