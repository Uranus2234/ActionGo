#include "action_go/runtime/activation_blend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace action_go::runtime {
namespace {

std::vector<float> lerp(const std::vector<float>& from,
                        const std::vector<float>& to,
                        double alpha) {
  std::vector<float> out(to.size());
  const float a = static_cast<float>(std::clamp(alpha, 0.0, 1.0));
  for (size_t i = 0; i < to.size(); ++i) out[i] = from[i] + (to[i] - from[i]) * a;
  return out;
}

double max_delta(const std::vector<float>& a, const std::vector<float>& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result,
                      std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return result;
}

}  // namespace

ActivationBlend::ActivationBlend(double duration_sec) {
  set_duration(duration_sec);
}

void ActivationBlend::set_duration(double duration_sec) {
  if (!std::isfinite(duration_sec) || duration_sec < 0.0 || duration_sec > 5.0) {
    throw std::invalid_argument("activation_blend_sec must be finite and within [0, 5]");
  }
  duration_sec_ = duration_sec;
  reset();
}

void ActivationBlend::validate_command(const JointCommand& command, const char* label) {
  if (command.q.empty() || command.q.size() != command.kp.size() ||
      command.q.size() != command.kd.size()) {
    throw std::invalid_argument(std::string(label) + " q/kp/kd dimensions must match");
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finite(command.q) || !finite(command.kp) || !finite(command.kd)) {
    throw std::invalid_argument(std::string(label) + " contains NaN/Inf");
  }
}

void ActivationBlend::observe_hold(const JointCommand& filtered_hold) {
  validate_command(filtered_hold, "activation hold command");
  if (started_) return;
  hold_ = filtered_hold;
  have_hold_ = true;
}

ActivationBlendResult ActivationBlend::apply(const JointCommand& active,
                                             Clock::time_point now,
                                             bool bypass) {
  validate_command(active, "activation active command");
  ActivationBlendResult result;
  result.command = active;
  if (bypass || duration_sec_ == 0.0 || !have_hold_) {
    result.bypassed = bypass;
    result.completed = !completed_;
    started_ = true;
    completed_ = true;
    return result;
  }
  if (active.q.size() != hold_.q.size()) {
    throw std::invalid_argument("activation hold/active command dimensions must match");
  }
  if (!started_) {
    start_ = now;
    started_ = true;
    result.started = true;
  }
  result.max_q_delta = max_delta(hold_.q, active.q);
  result.max_kp_delta = max_delta(hold_.kp, active.kp);
  result.max_kd_delta = max_delta(hold_.kd, active.kd);
  result.alpha = std::clamp(
      std::chrono::duration<double>(now - start_).count() / duration_sec_, 0.0, 1.0);
  result.command.q = lerp(hold_.q, active.q, result.alpha);
  result.command.kp = lerp(hold_.kp, active.kp, result.alpha);
  result.command.kd = lerp(hold_.kd, active.kd, result.alpha);
  if (result.alpha >= 1.0 && !completed_) {
    completed_ = true;
    result.completed = true;
  }
  return result;
}

void ActivationBlend::reset() {
  hold_ = {};
  have_hold_ = false;
  started_ = false;
  completed_ = false;
  start_ = {};
}

}  // namespace action_go::runtime
