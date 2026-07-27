#pragma once

#include "action_go/runtime/handoff_protocol.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace action_go::runtime {

enum class CommandArbiterState {
  kSourceActive,
  kTargetPreview,
  kTargetArmed,
  kCommit,
  kHoldRecovery,
};

enum class CommandSelection {
  kNone,
  kSource,
  kTarget,
  kHold,
};

const char* command_arbiter_state_name(CommandArbiterState state);
const char* command_selection_name(CommandSelection selection);

struct JointCommandCandidate {
  std::vector<float> q;
  std::vector<float> dq;
  std::vector<float> tau;
  std::vector<float> kp;
  std::vector<float> kd;
  int mode_machine{0};
  uint64_t sequence{0};
  std::chrono::steady_clock::time_point stamp{};
};

struct ArbiterRobotState {
  std::vector<float> q;
  std::vector<float> dq;
  uint64_t sequence{0};
  int mode_machine{0};
  bool connected{false};
  std::chrono::steady_clock::time_point stamp{};
};

struct CommandContinuityMetrics {
  double max_position_jump_rad{0.0};
  double max_velocity_jump{0.0};
  double max_torque_jump{0.0};
  double max_position_rate{0.0};
  double max_position_jerk{0.0};
  double source_age_sec{std::numeric_limits<double>::infinity()};
  double target_age_sec{std::numeric_limits<double>::infinity()};
};

struct CandidateCommandArbiterConfig {
  size_t motor_count{29};
  double candidate_timeout_sec{0.10};
  double lowstate_timeout_sec{0.20};
  float kp_max{300.f};
  float kd_max{20.f};
  float dq_abs_max{30.f};
  float tau_abs_max{200.f};
  float hold_kp{20.f};
  float hold_kd{1.f};
  // Zero disables rejection while metrics are still reported.
  float reject_target_position_jump_rad{1.2f};
  float reject_target_velocity_jump{20.f};
  float reject_target_torque_jump{120.f};
  bool require_mode_machine_match{true};
  SourceExitGateConfig source_exit_gate;
  std::vector<float> lower_limits;
  std::vector<float> upper_limits;
};

struct ArbiterOutput {
  bool publish{false};
  bool state_changed{false};
  bool fallback{false};
  CommandSelection selection{CommandSelection::kNone};
  CommandArbiterState state{CommandArbiterState::kSourceActive};
  JointCommandCandidate command;
  CommandContinuityMetrics metrics;
  HandoffAssessmentV1 assessment;
  std::string reason;
};

class CandidateCommandArbiter {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit CandidateCommandArbiter(CandidateCommandArbiterConfig config = {});

  using HrmRiskEvaluator =
      std::function<std::optional<double>(const HrmFeatureVectorV1&)>;

  void set_context(std::string run_id, std::string trial_id, std::string pair_id,
                   std::string source_policy, std::string target_policy);
  void update_lowstate(ArbiterRobotState state);
  void update_source(JointCommandCandidate command);
  void update_target(JointCommandCandidate command);
  void update_source_exit_status(SourceExitStatusV1 status);
  void set_hrm_risk_evaluator(HrmRiskEvaluator evaluator,
                              double authorization_threshold);

  void request_preview();
  void request_commit();
  void request_discard();

  CommandArbiterState state() const { return state_; }
  ArbiterOutput tick(TimePoint now = Clock::now());

 private:
  struct CandidateCheck {
    bool valid{false};
    JointCommandCandidate command;
    double age_sec{std::numeric_limits<double>::infinity()};
    std::string reason;
  };

  CandidateCheck check_candidate(const JointCommandCandidate& candidate,
                                 TimePoint now) const;
  bool lowstate_fresh(TimePoint now, std::string& reason) const;
  JointCommandCandidate make_hold(TimePoint now) const;
  JointCommandCandidate filter_command(JointCommandCandidate command) const;
  CommandContinuityMetrics continuity_metrics(const JointCommandCandidate& command,
                                               TimePoint now) const;
  void remember_output(const JointCommandCandidate& command, TimePoint now,
                       const CommandContinuityMetrics& metrics);
  HandoffAssessmentV1 assessment_for(const CandidateCheck& target,
                                     const CommandContinuityMetrics& metrics,
                                     bool authorized,
                                     const std::string& reason,
                                     HandoffDecision decision = HandoffDecision::kPending) const;
  HrmFeatureVectorV1 hrm_features(const CandidateCheck& target,
                                  const CommandContinuityMetrics& metrics) const;

  CandidateCommandArbiterConfig config_;
  CommandArbiterState state_{CommandArbiterState::kSourceActive};
  CommandArbiterState reported_state_{CommandArbiterState::kSourceActive};
  ArbiterRobotState lowstate_;
  JointCommandCandidate source_;
  JointCommandCandidate target_;
  SourceExitStatusV1 source_exit_status_;
  SourceExitGateResult source_exit_gate_result_;
  HrmRiskEvaluator hrm_risk_evaluator_;
  double hrm_authorization_threshold_{0.5};
  double latest_failure_probability_{std::numeric_limits<double>::quiet_NaN()};
  bool preview_requested_{false};
  bool commit_requested_{false};
  bool discard_requested_{false};
  JointCommandCandidate last_output_;
  std::vector<double> last_position_rate_;
  TimePoint last_output_time_{};
  std::string run_id_;
  std::string trial_id_;
  std::string pair_id_;
  std::string source_policy_;
  std::string target_policy_;
};

}  // namespace action_go::runtime
