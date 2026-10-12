#include "action_go/runtime/command_safety_filter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace action_go::runtime {
namespace {

float checked_finite(float value, const std::string& field, size_t index, bool enabled) {
  if (enabled && !std::isfinite(value)) {
    throw std::runtime_error("command safety rejected " + field + "[" + std::to_string(index) + "] NaN/Inf");
  }
  return value;
}

float safe_at(const std::vector<float>& values, size_t index, float fallback) {
  return index < values.size() ? values[index] : fallback;
}

}  // namespace

FilteredCommand CommandSafetyFilter::filter(const std::vector<float>& q,
                                            const std::vector<float>& kp,
                                            const std::vector<float>& kd,
                                            const CommandSafetyState& state,
                                            const CommandSafetyLimits& limits,
                                            size_t motor_count,
                                            bool allow_stale_state) const {
  if (!config_.enabled) {
    FilteredCommand out;
    out.q = q;
    out.kp = kp;
    out.kd = kd;
    out.q.resize(motor_count, config_.fallback_q);
    out.kp.resize(motor_count, 0.f);
    out.kd.resize(motor_count, 0.f);
    out.tau.resize(motor_count, 0.f);
    return out;
  }

  if (config_.lowstate_timeout_enabled && !allow_stale_state) {
    if (!state.connected || state.stamp == std::chrono::steady_clock::time_point{}) {
      throw std::runtime_error("command safety rejected publish before fresh LowState");
    }
    const auto now = std::chrono::steady_clock::now();
    const double age = std::chrono::duration<double>(now - state.stamp).count();
    if (age > config_.lowstate_timeout_sec) {
      throw std::runtime_error("command safety rejected stale LowState age=" + std::to_string(age) +
                               "s seq=" + std::to_string(state.seq));
    }
  }

  FilteredCommand out;
  out.q.resize(motor_count);
  out.kp.resize(motor_count);
  out.kd.resize(motor_count);
  out.tau.resize(motor_count, 0.f);

  for (size_t i = 0; i < motor_count; ++i) {
    float q_fallback = config_.fallback_q;
    if (config_.use_current_q_for_missing && i < state.q.size()) q_fallback = state.q[i];

    float qv = checked_finite(safe_at(q, i, q_fallback), "q", i, config_.finite_check);
    float kpv = checked_finite(safe_at(kp, i, 0.f), "kp", i, config_.finite_check);
    float kdv = checked_finite(safe_at(kd, i, 0.f), "kd", i, config_.finite_check);
    kpv = std::clamp(kpv, 0.f, std::max(0.f, config_.kp_max));
    kdv = std::clamp(kdv, 0.f, std::max(0.f, config_.kd_max));
    float desired_torque = 0.f;
    if (config_.preserve_pd_torque && !allow_stale_state) {
      if (i >= state.q.size() || i >= state.dq.size() || i >= limits.torque.size() ||
          !std::isfinite(state.q[i]) || !std::isfinite(state.dq[i]) ||
          !std::isfinite(limits.torque[i]) || limits.torque[i] <= 0.f) {
        throw std::runtime_error("PD torque filter requires finite q/dq and positive torque limits");
      }
      desired_torque = std::clamp(kpv * (qv - state.q[i]) - kdv * state.dq[i],
                                  -limits.torque[i], limits.torque[i]);
      if (kpv > 1e-6f) qv = state.q[i] + (desired_torque + kdv * state.dq[i]) / kpv;
    }

    if (config_.finite_check && i < state.q.size() && !std::isfinite(state.q[i])) {
      throw std::runtime_error("command safety rejected LowState q[" + std::to_string(i) + "] NaN/Inf");
    }

    if (config_.joint_limit_enabled && i < limits.lower.size() && i < limits.upper.size() &&
        std::isfinite(limits.lower[i]) && std::isfinite(limits.upper[i]) && limits.lower[i] <= limits.upper[i]) {
      qv = std::clamp(qv, limits.lower[i], limits.upper[i]);
    }

    if (config_.max_position_step_enabled && i < state.q.size() && std::isfinite(state.q[i])) {
      const float step = std::max(0.f, config_.max_position_step_rad);
      qv = std::clamp(qv, state.q[i] - step, state.q[i] + step);
    }

    if (config_.preserve_pd_torque && !allow_stale_state) {
      out.tau[i] = desired_torque - (kpv * (qv - state.q[i]) - kdv * state.dq[i]);
      if (std::abs(out.tau[i]) > limits.torque[i] + 1e-3f) {
        kpv = 0.f;
        kdv = 0.f;
        out.tau[i] = desired_torque;
      }
      checked_finite(out.tau[i], "tau", i, config_.finite_check);
    }

    out.q[i] = qv;
    out.kp[i] = kpv;
    out.kd[i] = kdv;
  }

  return out;
}

}  // namespace action_go::runtime
