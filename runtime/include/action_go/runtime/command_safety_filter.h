#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace action_go::runtime {

struct CommandSafetyConfig {
  bool enabled{true};
  bool finite_check{true};
  bool joint_limit_enabled{true};
  bool lowstate_timeout_enabled{true};
  double lowstate_timeout_sec{0.2};
  bool max_position_step_enabled{false};
  float max_position_step_rad{0.25f};
  float kp_max{300.f};
  float kd_max{20.f};
  bool use_current_q_for_missing{true};
  float fallback_q{0.f};
};

struct CommandSafetyState {
  bool connected{false};
  std::chrono::steady_clock::time_point stamp{};
  uint64_t seq{0};
  std::vector<float> q;
};

struct CommandSafetyLimits {
  std::vector<float> lower;
  std::vector<float> upper;
};

struct FilteredCommand {
  std::vector<float> q;
  std::vector<float> kp;
  std::vector<float> kd;
};

class CommandSafetyFilter {
 public:
  CommandSafetyFilter() = default;
  explicit CommandSafetyFilter(CommandSafetyConfig config) : config_(config) {}

  void set_config(CommandSafetyConfig config) { config_ = config; }
  const CommandSafetyConfig& config() const { return config_; }

  FilteredCommand filter(const std::vector<float>& q,
                         const std::vector<float>& kp,
                         const std::vector<float>& kd,
                         const CommandSafetyState& state,
                         const CommandSafetyLimits& limits,
                         size_t motor_count,
                         bool allow_stale_state = false) const;

 private:
  CommandSafetyConfig config_;
};

}  // namespace action_go::runtime
