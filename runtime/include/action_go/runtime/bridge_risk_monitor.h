#pragma once

#include "action_go/runtime/onnx_runner.h"

#include <array>
#include <chrono>
#include <deque>
#include <filesystem>
#include <vector>

namespace action_go::runtime {

struct BridgeRiskState {
  std::array<float, 29> q{};
  std::array<float, 29> dq{};
  std::array<float, 3> gyro{};
  std::array<float, 3> projected_gravity{};
  std::chrono::steady_clock::time_point stamp{};
};

class BridgeRiskMonitor {
 public:
  struct Config {
    std::filesystem::path model_path;
    int history_steps{10};
    int onnx_threads{2};
    std::string execution_provider{"cpu"};
    float stale_after_sec{0.20F};
    float risk_reference{36.4892061F};
  };

  explicit BridgeRiskMonitor(Config config);
  void load();
  bool loaded() const { return loaded_; }
  float update(const BridgeRiskState& state);
  float risk_percent() const { return risk_percent_; }
  bool valid() const { return valid_; }

 private:
  Config config_;
  OnnxRunner runner_;
  std::deque<std::array<float, 93>> history_;
  std::array<float, 29> previous_action_q_{};
  bool have_previous_action_{false};
  bool loaded_{false};
  bool valid_{false};
  float risk_percent_{100.0F};
};

}  // namespace action_go::runtime
