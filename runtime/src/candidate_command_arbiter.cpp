#include "action_go/runtime/candidate_command_arbiter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace action_go::runtime {
namespace {

double age_seconds(std::chrono::steady_clock::time_point now,
                   std::chrono::steady_clock::time_point stamp) {
  if (stamp == std::chrono::steady_clock::time_point{}) {
    return std::numeric_limits<double>::infinity();
  }
  return std::chrono::duration<double>(now - stamp).count();
}

float value_at(const std::vector<float>& values, size_t index, float fallback = 0.f) {
  return index < values.size() ? values[index] : fallback;
}

bool finite_vector(const std::vector<float>& values, size_t count) {
  if (values.size() < count) return false;
  for (size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) return false;
  }
  return true;
}

}  // namespace

const char* command_arbiter_state_name(CommandArbiterState state) {
  switch (state) {
    case CommandArbiterState::kSourceActive: return "source_active";
    case CommandArbiterState::kTargetPreview: return "target_preview";
    case CommandArbiterState::kTargetArmed: return "target_armed";
    case CommandArbiterState::kCommit: return "commit";
    case CommandArbiterState::kHoldRecovery: return "hold_recovery";
  }
  return "source_active";
}

const char* command_selection_name(CommandSelection selection) {
  switch (selection) {
    case CommandSelection::kNone: return "none";
    case CommandSelection::kSource: return "source";
    case CommandSelection::kTarget: return "target";
    case CommandSelection::kHold: return "hold";
  }
  return "none";
}

CandidateCommandArbiter::CandidateCommandArbiter(CandidateCommandArbiterConfig config)
    : config_(std::move(config)) {
  if (config_.motor_count == 0) throw std::invalid_argument("arbiter motor_count must be positive");
  if (config_.candidate_timeout_sec <= 0.0 || config_.lowstate_timeout_sec <= 0.0) {
    throw std::invalid_argument("arbiter timeouts must be positive");
  }
  last_position_rate_.assign(config_.motor_count, 0.0);
}

void CandidateCommandArbiter::set_context(std::string run_id, std::string trial_id,
                                          std::string pair_id, std::string source_policy,
                                          std::string target_policy) {
  run_id_ = std::move(run_id);
  trial_id_ = std::move(trial_id);
  pair_id_ = std::move(pair_id);
  source_policy_ = std::move(source_policy);
  target_policy_ = std::move(target_policy);
}

void CandidateCommandArbiter::update_lowstate(ArbiterRobotState state) {
  lowstate_ = std::move(state);
}

void CandidateCommandArbiter::update_source(JointCommandCandidate command) {
  source_ = std::move(command);
}

void CandidateCommandArbiter::update_target(JointCommandCandidate command) {
  target_ = std::move(command);
}

void CandidateCommandArbiter::update_source_exit_status(SourceExitStatusV1 status) {
  source_exit_status_ = std::move(status);
}

void CandidateCommandArbiter::set_hrm_risk_evaluator(
    HrmRiskEvaluator evaluator, double authorization_threshold) {
  if (!std::isfinite(authorization_threshold) || authorization_threshold < 0.0 ||
      authorization_threshold > 1.0) {
    throw std::invalid_argument("HRM authorization threshold must be in [0, 1]");
  }
  hrm_risk_evaluator_ = std::move(evaluator);
  hrm_authorization_threshold_ = authorization_threshold;
}

void CandidateCommandArbiter::request_preview() {
  preview_requested_ = true;
}

void CandidateCommandArbiter::request_commit() {
  commit_requested_ = true;
}

void CandidateCommandArbiter::request_discard() {
  discard_requested_ = true;
}

CandidateCommandArbiter::CandidateCheck CandidateCommandArbiter::check_candidate(
    const JointCommandCandidate& candidate, TimePoint now) const {
  CandidateCheck check;
  check.age_sec = age_seconds(now, candidate.stamp);
  if (check.age_sec < -1e-3) {
    check.reason = "candidate timestamp is in the future";
    return check;
  }
  if (check.age_sec > config_.candidate_timeout_sec) {
    check.reason = "candidate stale";
    return check;
  }
  if (!finite_vector(candidate.q, config_.motor_count) ||
      !finite_vector(candidate.dq, config_.motor_count) ||
      !finite_vector(candidate.tau, config_.motor_count) ||
      !finite_vector(candidate.kp, config_.motor_count) ||
      !finite_vector(candidate.kd, config_.motor_count)) {
    check.reason = "candidate dimension or finite check failed";
    return check;
  }
  for (size_t index = 0; index < config_.motor_count; ++index) {
    if (std::abs(candidate.dq[index]) > config_.dq_abs_max) {
      check.reason = "candidate velocity bound failed";
      return check;
    }
    if (std::abs(candidate.tau[index]) > config_.tau_abs_max) {
      check.reason = "candidate torque bound failed";
      return check;
    }
  }
  if (config_.require_mode_machine_match && lowstate_.mode_machine > 0 &&
      candidate.mode_machine > 0 && candidate.mode_machine != lowstate_.mode_machine) {
    check.reason = "candidate mode-machine mismatch";
    return check;
  }
  check.command = filter_command(candidate);
  check.valid = true;
  check.reason = "ok";
  return check;
}

bool CandidateCommandArbiter::lowstate_fresh(TimePoint now, std::string& reason) const {
  if (!lowstate_.connected || lowstate_.q.size() < config_.motor_count) {
    reason = "lowstate unavailable";
    return false;
  }
  const double age = age_seconds(now, lowstate_.stamp);
  if (age < -1e-3 || age > config_.lowstate_timeout_sec) {
    reason = "lowstate stale";
    return false;
  }
  if (!finite_vector(lowstate_.q, config_.motor_count)) {
    reason = "lowstate finite check failed";
    return false;
  }
  reason.clear();
  return true;
}

JointCommandCandidate CandidateCommandArbiter::make_hold(TimePoint now) const {
  JointCommandCandidate hold;
  hold.q.assign(lowstate_.q.begin(), lowstate_.q.begin() +
                                      static_cast<std::ptrdiff_t>(config_.motor_count));
  hold.dq.assign(config_.motor_count, 0.f);
  hold.tau.assign(config_.motor_count, 0.f);
  hold.kp.assign(config_.motor_count, config_.hold_kp);
  hold.kd.assign(config_.motor_count, config_.hold_kd);
  hold.mode_machine = lowstate_.mode_machine;
  hold.stamp = now;
  hold.sequence = lowstate_.sequence;
  return filter_command(std::move(hold));
}

JointCommandCandidate CandidateCommandArbiter::filter_command(
    JointCommandCandidate command) const {
  command.q.resize(config_.motor_count, 0.f);
  command.dq.resize(config_.motor_count, 0.f);
  command.tau.resize(config_.motor_count, 0.f);
  command.kp.resize(config_.motor_count, 0.f);
  command.kd.resize(config_.motor_count, 0.f);
  for (size_t index = 0; index < config_.motor_count; ++index) {
    if (index < config_.lower_limits.size() && index < config_.upper_limits.size() &&
        std::isfinite(config_.lower_limits[index]) &&
        std::isfinite(config_.upper_limits[index]) &&
        config_.lower_limits[index] <= config_.upper_limits[index]) {
      command.q[index] = std::clamp(command.q[index], config_.lower_limits[index],
                                    config_.upper_limits[index]);
    }
    command.kp[index] = std::clamp(command.kp[index], 0.f, std::max(0.f, config_.kp_max));
    command.kd[index] = std::clamp(command.kd[index], 0.f, std::max(0.f, config_.kd_max));
  }
  return command;
}

CommandContinuityMetrics CandidateCommandArbiter::continuity_metrics(
    const JointCommandCandidate& command, TimePoint now) const {
  CommandContinuityMetrics metrics;
  metrics.source_age_sec = age_seconds(now, source_.stamp);
  metrics.target_age_sec = age_seconds(now, target_.stamp);
  const std::vector<float>* reference_q = nullptr;
  const std::vector<float>* reference_dq = nullptr;
  const std::vector<float>* reference_tau = nullptr;
  TimePoint reference_time{};
  if (last_output_.q.size() >= config_.motor_count) {
    reference_q = &last_output_.q;
    reference_dq = &last_output_.dq;
    reference_tau = &last_output_.tau;
    reference_time = last_output_time_;
  } else if (lowstate_.q.size() >= config_.motor_count) {
    reference_q = &lowstate_.q;
    reference_dq = &lowstate_.dq;
  }
  if (!reference_q) return metrics;

  double dt = 0.0;
  if (reference_time != TimePoint{}) dt = std::chrono::duration<double>(now - reference_time).count();
  for (size_t index = 0; index < config_.motor_count; ++index) {
    const double q_delta = std::abs(static_cast<double>(command.q[index]) -
                                    static_cast<double>(value_at(*reference_q, index)));
    const double dq_delta = std::abs(static_cast<double>(command.dq[index]) -
                                     static_cast<double>(reference_dq ? value_at(*reference_dq, index) : 0.f));
    const double tau_delta = std::abs(static_cast<double>(command.tau[index]) -
                                      static_cast<double>(reference_tau ? value_at(*reference_tau, index) : 0.f));
    metrics.max_position_jump_rad = std::max(metrics.max_position_jump_rad, q_delta);
    metrics.max_velocity_jump = std::max(metrics.max_velocity_jump, dq_delta);
    metrics.max_torque_jump = std::max(metrics.max_torque_jump, tau_delta);
    if (dt > 1e-6) {
      const double rate = (static_cast<double>(command.q[index]) -
                           static_cast<double>(value_at(*reference_q, index))) / dt;
      metrics.max_position_rate = std::max(metrics.max_position_rate, std::abs(rate));
      if (index < last_position_rate_.size()) {
        metrics.max_position_jerk = std::max(
            metrics.max_position_jerk, std::abs(rate - last_position_rate_[index]) / dt);
      }
    }
  }
  return metrics;
}

void CandidateCommandArbiter::remember_output(const JointCommandCandidate& command,
                                              TimePoint now,
                                              const CommandContinuityMetrics&) {
  if (last_output_.q.size() >= config_.motor_count && last_output_time_ != TimePoint{}) {
    const double dt = std::chrono::duration<double>(now - last_output_time_).count();
    if (dt > 1e-6) {
      for (size_t index = 0; index < config_.motor_count; ++index) {
        last_position_rate_[index] =
            (static_cast<double>(command.q[index]) - last_output_.q[index]) / dt;
      }
    }
  }
  last_output_ = command;
  last_output_time_ = now;
}

HandoffAssessmentV1 CandidateCommandArbiter::assessment_for(
    const CandidateCheck& target, const CommandContinuityMetrics& metrics,
    bool authorized, const std::string& reason, HandoffDecision decision) const {
  HandoffAssessmentV1 assessment;
  assessment.run_id = run_id_;
  assessment.trial_id = trial_id_;
  assessment.pair_id = pair_id_;
  assessment.source_policy = source_policy_;
  assessment.target_policy = target_policy_;
  assessment.group = pair_id_;
  assessment.source_sequence = source_.sequence;
  assessment.target_sequence = target.command.sequence;
  assessment.decision = decision;
  assessment.source_phase = source_exit_status_.phase;
  assessment.source_exit_admissible = source_exit_gate_result_.admissible;
  assessment.target_viability = target.valid;
  assessment.retry_policy = source_exit_status_.retry_policy;
  if (!authorized) assessment.reject_reason = reason;
  assessment.hard_preconditions_met = source_exit_gate_result_.admissible && target.valid;
  assessment.authorized = authorized;
  assessment.failure_probability = latest_failure_probability_;
  assessment.target_age_sec = target.age_sec;
  assessment.max_position_jump_rad = metrics.max_position_jump_rad;
  assessment.max_velocity_jump = metrics.max_velocity_jump;
  assessment.max_torque_jump = metrics.max_torque_jump;
  assessment.reason = reason;
  return assessment;
}

HrmFeatureVectorV1 CandidateCommandArbiter::hrm_features(
    const CandidateCheck& target, const CommandContinuityMetrics& metrics) const {
  HrmFeatureVectorV1 features;
  features.names = {
      "source_contact_count", "source_airborne", "source_rotation",
      "source_fallen", "source_base_height_m", "source_tilt_deg",
      "source_angular_speed_rps", "target_age_sec",
      "max_position_jump_rad", "max_velocity_jump", "max_torque_jump",
      "max_position_rate", "max_position_jerk", "mode_machine_match",
      "target_valid", "source_exit_admissible"};
  const bool mode_match = target.command.mode_machine <= 0 || lowstate_.mode_machine <= 0 ||
      target.command.mode_machine == lowstate_.mode_machine;
  auto finite_or_zero = [](double value) {
    return static_cast<float>(std::isfinite(value) ? value : 0.0);
  };
  features.values = {
      static_cast<float>(source_exit_status_.contact_count),
      source_exit_status_.airborne ? 1.f : 0.f,
      source_exit_status_.rotation ? 1.f : 0.f,
      source_exit_status_.fallen ? 1.f : 0.f,
      finite_or_zero(source_exit_status_.base_height_m),
      finite_or_zero(source_exit_status_.tilt_deg),
      finite_or_zero(source_exit_status_.angular_speed_rps),
      finite_or_zero(target.age_sec),
      finite_or_zero(metrics.max_position_jump_rad),
      finite_or_zero(metrics.max_velocity_jump),
      finite_or_zero(metrics.max_torque_jump),
      finite_or_zero(metrics.max_position_rate),
      finite_or_zero(metrics.max_position_jerk),
      mode_match ? 1.f : 0.f,
      target.valid ? 1.f : 0.f,
      source_exit_gate_result_.admissible ? 1.f : 0.f};
  return features;
}

ArbiterOutput CandidateCommandArbiter::tick(TimePoint now) {
  ArbiterOutput output;
  std::string lowstate_reason;
  if (!lowstate_fresh(now, lowstate_reason)) {
    output.state = state_;
    output.reason = lowstate_reason;
    output.assessment = assessment_for({}, {}, false, lowstate_reason);
    return output;
  }

  source_exit_gate_result_ = evaluate_source_exit_gate(
      source_exit_status_, config_.source_exit_gate);

  if (discard_requested_) {
    state_ = CommandArbiterState::kSourceActive;
    discard_requested_ = false;
    preview_requested_ = false;
    commit_requested_ = false;
    latest_failure_probability_ = std::numeric_limits<double>::quiet_NaN();
  }
  if (preview_requested_ && state_ == CommandArbiterState::kSourceActive) {
    if (source_exit_gate_result_.admissible) {
      state_ = CommandArbiterState::kTargetPreview;
    }
    preview_requested_ = false;
  }

  const CandidateCheck source = check_candidate(source_, now);
  const CandidateCheck target = check_candidate(target_, now);
  CommandContinuityMetrics target_metrics;
  if (target.valid) target_metrics = continuity_metrics(target.command, now);

  if (!source_exit_gate_result_.admissible &&
      state_ != CommandArbiterState::kCommit) {
    state_ = CommandArbiterState::kSourceActive;
    commit_requested_ = false;
    output.assessment = assessment_for(
        target, target_metrics, false, source_exit_gate_result_.reason,
        source_exit_gate_result_.decision);
    output.reason = source_exit_gate_result_.reason;
  }

  if (state_ == CommandArbiterState::kTargetPreview) {
    const bool jump_ok = config_.reject_target_position_jump_rad <= 0.f ||
        target_metrics.max_position_jump_rad <= config_.reject_target_position_jump_rad;
    const bool velocity_ok = config_.reject_target_velocity_jump <= 0.f ||
        target_metrics.max_velocity_jump <= config_.reject_target_velocity_jump;
    const bool torque_ok = config_.reject_target_torque_jump <= 0.f ||
        target_metrics.max_torque_jump <= config_.reject_target_torque_jump;
    bool hrm_ok = true;
    latest_failure_probability_ = std::numeric_limits<double>::quiet_NaN();
    if (target.valid && jump_ok && velocity_ok && torque_ok && hrm_risk_evaluator_) {
      const auto probability = hrm_risk_evaluator_(hrm_features(target, target_metrics));
      if (!probability || !std::isfinite(*probability) ||
          *probability < 0.0 || *probability > 1.0) {
        hrm_ok = false;
      } else {
        latest_failure_probability_ = *probability;
        hrm_ok = *probability <= hrm_authorization_threshold_;
      }
    }
    if (target.valid && jump_ok && velocity_ok && torque_ok && hrm_ok) {
      state_ = CommandArbiterState::kTargetArmed;
    } else {
      std::string reject_reason = target.reason;
      if (target.valid && !jump_ok) reject_reason = "target position jump rejected";
      else if (target.valid && !velocity_ok) reject_reason = "target velocity jump rejected";
      else if (target.valid && !torque_ok) reject_reason = "target torque jump rejected";
      else if (target.valid && !hrm_ok && std::isfinite(latest_failure_probability_)) {
        reject_reason = "HRM risk exceeds authorization threshold";
      } else if (target.valid && !hrm_ok) {
        reject_reason = "HRM inference unavailable or invalid";
      }
      output.assessment = assessment_for(
          target, target_metrics, false, reject_reason,
          HandoffDecision::kPending);
    }
  }

  if (commit_requested_) {
    commit_requested_ = false;
    if (state_ == CommandArbiterState::kTargetArmed && target.valid) {
      state_ = CommandArbiterState::kCommit;
    }
  }

  if (state_ == CommandArbiterState::kCommit && !target.valid) {
    state_ = CommandArbiterState::kHoldRecovery;
    output.fallback = true;
    output.reason = "target candidate lost after commit; hold/recovery required: " + target.reason;
    output.assessment = assessment_for(
        target, target_metrics, false, output.reason, HandoffDecision::kHold);
  }

  output.state_changed = state_ != reported_state_;
  reported_state_ = state_;
  output.state = state_;

  if (state_ == CommandArbiterState::kCommit && target.valid) {
    output.publish = true;
    output.selection = CommandSelection::kTarget;
    output.command = target.command;
    output.metrics = continuity_metrics(output.command, now);
    output.assessment = assessment_for(
        target, output.metrics, true, "commit", HandoffDecision::kDirect);
    output.reason = "target committed";
  } else if (state_ == CommandArbiterState::kHoldRecovery) {
    output.publish = true;
    output.selection = CommandSelection::kHold;
    output.command = make_hold(now);
    output.metrics = continuity_metrics(output.command, now);
    output.fallback = true;
    if (output.reason.empty()) {
      output.reason = "post-commit hold/recovery state";
      output.assessment = assessment_for(
          target, target_metrics, false, output.reason, HandoffDecision::kHold);
    }
  } else if (source.valid) {
    output.publish = true;
    output.selection = CommandSelection::kSource;
    output.command = source.command;
    output.metrics = continuity_metrics(output.command, now);
    if (output.assessment.reason.empty()) {
      const bool armed = state_ == CommandArbiterState::kTargetArmed;
      output.assessment = assessment_for(
          target, target_metrics, armed, armed ? "target armed" : "source retained",
          armed ? HandoffDecision::kDirect : HandoffDecision::kPending);
    }
    if (output.reason.empty()) output.reason = "source active";
  } else {
    output.publish = true;
    output.selection = CommandSelection::kHold;
    output.command = make_hold(now);
    output.metrics = continuity_metrics(output.command, now);
    output.fallback = true;
    output.reason = "source unavailable; publishing current-pose hold";
    if (output.assessment.reason.empty()) {
      output.assessment = assessment_for(target, target_metrics, false, source.reason);
    }
  }

  remember_output(output.command, now, output.metrics);
  return output;
}

}  // namespace action_go::runtime
