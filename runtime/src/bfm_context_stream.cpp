#include "action_go/runtime/bfm_context_stream.h"
#include "action_go/runtime/reference_endpoint.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <stdexcept>

namespace action_go::runtime {
namespace {

double unix_time() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}

BfmContextStream::BfmContextStream(const std::string& bind_address, int port,
                                 double timeout_sec, size_t latent_dim)
    : timeout_sec_(timeout_sec), latent_dim_(latent_dim) {
  if (!is_local_reference_bind(bind_address)) {
    throw std::runtime_error("BFM context UDP bind address must be loopback");
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (port <= 0 || port > 65535 || !std::isfinite(timeout_sec) || timeout_sec <= 0.0 ||
      inet_pton(AF_INET, bind_address.c_str(), &address.sin_addr) != 1) {
    throw std::runtime_error("invalid BFM realtime UDP configuration");
  }
  socket_ = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (socket_ < 0) throw std::runtime_error("cannot create BFM realtime UDP socket");
  if (bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    close(socket_);
    socket_ = -1;
    throw std::runtime_error("cannot bind BFM realtime UDP port " + std::to_string(port));
  }
}

BfmContextStream::~BfmContextStream() {
  if (socket_ >= 0) close(socket_);
}

void BfmContextStream::poll() {
  std::array<char, 16384> buffer{};
  for (size_t packet = 0; packet < 256; ++packet) {
    const auto count = recv(socket_, buffer.data(), buffer.size(), MSG_DONTWAIT | MSG_TRUNC);
    if (count < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return;
      if (errno == EINTR) continue;
      throw std::runtime_error("BFM realtime UDP receive failed");
    }
    if (static_cast<size_t>(count) >= buffer.size()) continue;
    try {
    const auto message = nlohmann::json::parse(buffer.data(), buffer.data() + count);
    if (message.at("contract") != "bfm_realtime_context_v1") {
      throw std::runtime_error("unsupported BFM realtime context contract");
    }
    const auto sequence = message.at("sequence").get<int64_t>();
    const auto capture = message.at("capture_unix_sec").get<double>();
    if (sequence <= sequence_ || (has_sample_ && capture <= capture_time_)) continue;
    const auto age = unix_time() - capture;
    if (!std::isfinite(capture) || age < -timeout_sec_) {
      throw std::runtime_error("BFM realtime source timestamp is stale or clocks are not synchronized");
    }
    if (age > timeout_sec_) continue;
    const auto context = BfmContext::from_yaml(YAML::Load(message.dump()), latent_dim_);
    float norm_squared = 0.f;
    for (float value : context.latent) norm_squared += value * value;
    if (std::abs(std::sqrt(norm_squared) - std::sqrt(static_cast<float>(latent_dim_))) > 0.01f) {
      throw std::runtime_error("BFM realtime latent norm differs from training projection");
    }
    context_ = context;
    capture_time_ = capture;
    sequence_ = sequence;
    has_sample_ = true;
    received_at_ = std::chrono::steady_clock::now();
    } catch (const std::exception&) {
      continue;
    }
  }
}

BfmContext BfmContextStream::current() {
  poll();
  if (!has_sample_) throw std::runtime_error("BFM realtime context is missing");
  if (std::chrono::duration<double>(std::chrono::steady_clock::now() - received_at_).count() > timeout_sec_ ||
      unix_time() - capture_time_ > timeout_sec_) {
    throw std::runtime_error("BFM realtime context is stale");
  }
  return context_;
}

}
