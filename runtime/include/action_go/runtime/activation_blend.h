#pragma once

#include <chrono>
#include <cstdint>
#include <vector>
#include "action_go/runtime/command_timing.h"

namespace action_go::runtime {

struct JointCommand {
  std::vector<float> q;
  std::vector<float> kp;
  std::vector<float> kd;
  uint64_t sequence{0};
  CommandTiming timing;
};

struct ActivationBlendResult {
  JointCommand command;
  double alpha{1.0};
  bool started{false};
  bool completed{false};
  bool bypassed{false};
  double max_q_delta{0.0};
  double max_kp_delta{0.0};
  double max_kd_delta{0.0};
};

class ActivationBlend {
 public:
  using Clock = std::chrono::steady_clock;

  explicit ActivationBlend(double duration_sec = 0.10);

  void set_duration(double duration_sec);
  double duration_sec() const { return duration_sec_; }

  void observe_hold(const JointCommand& filtered_hold);
  ActivationBlendResult apply(const JointCommand& active,
                              Clock::time_point now = Clock::now(),
                              bool bypass = false);
  void reset();

 private:
  static void validate_command(const JointCommand& command, const char* label);

  double duration_sec_{0.10};
  JointCommand hold_;
  bool have_hold_{false};
  bool started_{false};
  bool completed_{false};
  Clock::time_point start_{};
};

}  // namespace action_go::runtime
