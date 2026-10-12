#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace action_go::runtime {

class DepthMonitorPublisher {
 public:
  explicit DepthMonitorPublisher(const std::string& path) {
    if (path.empty()) return;
    if (path.size() >= sizeof(address_.sun_path)) {
      std::cerr << "[depth monitor] socket path too long; monitor disabled\n";
      return;
    }
    address_.sun_family = AF_UNIX;
    std::memcpy(address_.sun_path, path.c_str(), path.size() + 1);
    descriptor_ = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (descriptor_ < 0) {
      std::cerr << "[depth monitor] cannot create socket; monitor disabled\n";
    }
  }

  ~DepthMonitorPublisher() {
    if (descriptor_ >= 0) close(descriptor_);
  }

  DepthMonitorPublisher(const DepthMonitorPublisher&) = delete;
  DepthMonitorPublisher& operator=(const DepthMonitorPublisher&) = delete;

  void publish(const std::vector<float>& depth, const std::array<float, 4>& quaternion,
               bool torso_imu) {
    if (descriptor_ < 0 || depth.size() != 58 * 87) return;
    const auto now = std::chrono::steady_clock::now();
    if (now < next_send_) return;
    next_send_ = now + std::chrono::milliseconds(100);
    std::array<unsigned char, 36> header{};
    std::memcpy(header.data(), "AGDEPTH1", 8);
    const uint64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const uint32_t source = torso_imu ? 1 : 0;
    std::memcpy(header.data() + 8, &timestamp, sizeof(timestamp));
    std::memcpy(header.data() + 16, &source, sizeof(source));
    std::memcpy(header.data() + 20, quaternion.data(), 4 * sizeof(float));
    std::array<iovec, 2> buffers{};
    buffers[0] = {header.data(), header.size()};
    buffers[1] = {const_cast<float*>(depth.data()), depth.size() * sizeof(float)};
    msghdr message{};
    message.msg_name = &address_;
    message.msg_namelen = sizeof(address_);
    message.msg_iov = buffers.data();
    message.msg_iovlen = buffers.size();
    sendmsg(descriptor_, &message, MSG_DONTWAIT | MSG_NOSIGNAL);
  }

 private:
  int descriptor_{-1};
  sockaddr_un address_{};
  std::chrono::steady_clock::time_point next_send_{};
};

}
