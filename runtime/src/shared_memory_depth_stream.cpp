#include "action_go/runtime/shared_memory_depth_stream.h"
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <zmq.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace action_go::runtime {

class SharedMemoryDepthStream::Impl {
 public:
  Impl(const std::string& shm_name, size_t pixel_count, double timeout_sec)
      : size_(pixel_count) {
    descriptor_ = shm_open(("/" + shm_name).c_str(), O_RDONLY, 0);
    if (descriptor_ < 0) throw std::runtime_error("depth shared memory missing: " + shm_name);
    struct stat status{};
    if (fstat(descriptor_, &status) != 0 || status.st_size != static_cast<off_t>(size_ * sizeof(float))) {
      close(descriptor_);
      throw std::runtime_error("depth shared memory size mismatch");
    }
    mapping_ = mmap(nullptr, size_ * sizeof(float), PROT_READ, MAP_SHARED, descriptor_, 0);
    if (mapping_ == MAP_FAILED) {
      close(descriptor_);
      throw std::runtime_error("cannot map depth shared memory");
    }
    context_ = zmq_ctx_new();
    socket_ = zmq_socket(context_, ZMQ_SUB);
    const int timeout_ms = static_cast<int>(timeout_sec * 1000);
    const int conflate = 1;
    zmq_setsockopt(socket_, ZMQ_CONFLATE, &conflate, sizeof(conflate));
    zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    zmq_setsockopt(socket_, ZMQ_SUBSCRIBE, "", 0);
    const std::string endpoint = "ipc:///tmp/" + shm_name + ".heartbeat";
    zmq_connect(socket_, endpoint.c_str());
    try {
      if (!receive(0)) throw std::runtime_error("depth heartbeat unavailable before ready");
    } catch (...) {
      release();
      throw;
    }
  }
  ~Impl() { release(); }
  std::vector<float> observation() {
    while (receive(ZMQ_DONTWAIT)) {}
    if (failed_ || age_sec() > 0.5) throw std::runtime_error("depth producer stopped or stale");
    const auto* data = static_cast<const float*>(mapping_);
    std::vector<float> frame(data, data + size_);
    for (float value : frame) if (!std::isfinite(value)) throw std::runtime_error("non-finite depth input");
    return frame;
  }
  int frame_count() const { return static_cast<int>(sequence_); }
  uint64_t sequence() const { return sequence_; }
  double age_sec() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - stamp_).count();
  }
  std::string stats() const { return "source=shm seq=" + std::to_string(sequence_) + " age=" + std::to_string(age_sec()); }
 private:
  bool receive(int flags) {
    std::array<char,512> message{};
    const int count = zmq_recv(socket_, message.data(), message.size(), flags);
    if (count <= 0) return false;
    const auto heartbeat = YAML::Load(std::string(message.data(), static_cast<size_t>(count)));
    const auto sequence = heartbeat["seq"].as<uint64_t>();
    if (sequence_ > 0 && sequence < sequence_) throw std::runtime_error("depth producer restarted; restart player");
    if (sequence != sequence_) stamp_ = std::chrono::steady_clock::now();
    sequence_ = sequence;
    failed_ = heartbeat["state"].as<std::string>() != "OK";
    return true;
  }
  void release() {
    if (socket_) zmq_close(socket_);
    if (context_) zmq_ctx_term(context_);
    if (mapping_ && mapping_ != MAP_FAILED) munmap(mapping_, size_ * sizeof(float));
    if (descriptor_ >= 0) close(descriptor_);
  }
  size_t size_;
  int descriptor_{-1};
  void* mapping_{nullptr};
  void* context_{nullptr};
  void* socket_{nullptr};
  uint64_t sequence_{0};
  bool failed_{false};
  std::chrono::steady_clock::time_point stamp_{};
};

SharedMemoryDepthStream::SharedMemoryDepthStream(const std::string& name, size_t pixel_count, double timeout_sec)
    : impl_(std::make_unique<Impl>(name, pixel_count, timeout_sec)) {}
SharedMemoryDepthStream::~SharedMemoryDepthStream() = default;
std::vector<float> SharedMemoryDepthStream::observation() { return impl_->observation(); }
uint64_t SharedMemoryDepthStream::sequence() const { return impl_->sequence(); }
double SharedMemoryDepthStream::age_sec() const { return impl_->age_sec(); }
std::string SharedMemoryDepthStream::stats() const { return impl_->stats(); }

}
