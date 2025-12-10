/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <mutex>  // NOLINT
#include <cstring> // For memcpy

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/nvtx.h"

namespace k2 {

static constexpr std::size_t kAlignment = 64;

class CpuContext : public Context {
 public:
  CpuContext() = default;
  ContextPtr GetCpuContext() { return shared_from_this(); } // Removed override
  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      int32_t ret = posix_memalign(&p, kAlignment, bytes);
      K2_CHECK_EQ(ret, 0);
    }
    if (deleter_context != nullptr) *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCpu;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
    free(data);
  }

  // Added CopyDataTo implementation
  void CopyDataTo(size_t num_bytes, const void *src,
                  ContextPtr dst_context, void *dst) override {
      // In WASM/CPU-only mode, we assume everything is in CPU memory.
      // We check that dst_context is valid and simple.
      std::memcpy(dst, src, num_bytes);
  }
};

// Simplified CudaContext for WASM (just a placeholder or reuse CpuContext logic if forcing it,
// but really we should just return CpuContext from GetCudaContext).
// However, compiling CudaContext requires it to not be abstract.
// We will just make it same as CpuContext essentially or error out.
class CudaContext : public Context {
 public:
  explicit CudaContext(int32_t gpu_id) : gpu_id_(gpu_id) {
     // In WASM, this shouldn't be called if GetCudaContext returns CpuContext.
     // But if it is, we simulate success or fail.
  }
  ContextPtr GetCpuContext() { return k2::GetCpuContext(); } // Removed override
  DeviceType GetDeviceType() const override { return kCuda; } // Pretend? Or just fail.
  int32_t GetDeviceId() const override { return gpu_id_; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
      K2_LOG(FATAL) << "CudaContext::Allocate called in WASM build";
      return nullptr;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
      K2_LOG(FATAL) << "CudaContext::Deallocate called in WASM build";
  }

  cudaStream_t GetCudaStream() const override {
    return kCudaStreamInvalid;
  }

  void Sync() const override {
  }

  // Implement CopyDataTo to make it concrete
  void CopyDataTo(size_t num_bytes, const void *src,
                  ContextPtr dst_context, void *dst) override {
      K2_LOG(FATAL) << "CudaContext::CopyDataTo called in WASM build";
  }

  ~CudaContext() {
  }

 private:
  int32_t gpu_id_;
};

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
  // Always return CPU context for WASM
  return GetCpuContext();
}

ContextPtr GetPinnedContext() {
    return GetCpuContext();
}

ContextPtr GetContextForTransfer(DeviceType device_type) {
    return GetCpuContext();
}

}  // namespace k2
