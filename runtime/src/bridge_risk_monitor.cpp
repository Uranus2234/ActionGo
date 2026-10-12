#include "action_go/runtime/bridge_risk_monitor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace action_go::runtime {

BridgeRiskMonitor::BridgeRiskMonitor(Config config) : config_(std::move(config)) {
  if (config_.history_steps != 10 || !(config_.risk_reference > 0.0F)) {
    throw std::invalid_argument("BridgeRiskMonitor requires history_steps=10 and positive reference");
  }
}

void BridgeRiskMonitor::load() {
  if (config_.model_path.empty()) throw std::invalid_argument("risk model path is empty");
  OnnxRunner::LoadOptions options;
  options.threads = std::max(1, config_.onnx_threads);
  options.execution_provider = config_.execution_provider;
  runner_.load(config_.model_path, options);
  runner_.require_inputs({"history"});
  runner_.require_outputs({"risk_percent"});
  loaded_ = true;
  valid_ = false;
  history_.clear();
  previous_action_q_.fill(0.0F);
  have_previous_action_ = false;
}

float BridgeRiskMonitor::update(const BridgeRiskState& state) {
  valid_ = false;
  if (!loaded_ || state.stamp.time_since_epoch().count() == 0 ||
      std::chrono::steady_clock::now() - state.stamp >
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<float>(config_.stale_after_sec))) {
    risk_percent_ = 100.0F;
    return risk_percent_;
  }
  std::array<float, 93> frame{};
  std::copy(state.q.begin(), state.q.end(), frame.begin());
  std::copy(state.dq.begin(), state.dq.end(), frame.begin() + 29);
  std::copy(state.gyro.begin(), state.gyro.end(), frame.begin() + 58);
  std::copy(state.projected_gravity.begin(), state.projected_gravity.end(), frame.begin() + 61);
  const auto& previous = have_previous_action_ ? previous_action_q_ : state.q;
  std::copy(previous.begin(), previous.end(), frame.begin() + 64);
  if (!std::all_of(frame.begin(), frame.end(), [](float value) { return std::isfinite(value); })) {
    risk_percent_ = 100.0F;
    return risk_percent_;
  }
  history_.push_back(frame);
  previous_action_q_ = state.q;
  have_previous_action_ = true;
  while (history_.size() < static_cast<size_t>(config_.history_steps)) history_.push_front(frame);
  while (history_.size() > static_cast<size_t>(config_.history_steps)) history_.pop_front();
  std::vector<float> input;
  input.reserve(history_.size() * frame.size());
  for (const auto& sample : history_) input.insert(input.end(), sample.begin(), sample.end());
  const auto output = runner_.run_named(
      {{"history", OnnxRunner::TensorInput(std::move(input), {1, 10, 93})}},
      {{"risk_percent", 1}});
  const auto it = output.find("risk_percent");
  if (it == output.end() || it->second.size() != 1 || !std::isfinite(it->second[0])) {
    risk_percent_ = 100.0F;
    return risk_percent_;
  }
  risk_percent_ = std::clamp(it->second[0], 0.0F, 100.0F);
  valid_ = true;
  return risk_percent_;
}

}  // namespace action_go::runtime
