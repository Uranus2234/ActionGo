#pragma once

#include "action_go/runtime/bfm_context.h"

#include <chrono>
#include <cstdint>
#include <string>

namespace action_go::runtime {

class BfmContextStream {
 public:
  BfmContextStream(const std::string& bind_address, int port, double timeout_sec, size_t latent_dim);
  ~BfmContextStream();
  BfmContextStream(const BfmContextStream&) = delete;
  BfmContextStream& operator=(const BfmContextStream&) = delete;
  void poll();
  bool has_sample() const { return has_sample_; }
  BfmContext current();
  int64_t sequence() const { return sequence_; }

 private:
  int socket_{-1};
  double timeout_sec_;
  size_t latent_dim_;
  bool has_sample_{false};
  int64_t sequence_{-1};
  double capture_time_{0.0};
  std::chrono::steady_clock::time_point received_at_;
  BfmContext context_;
};

}
