#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace action_go::runtime {

constexpr int kPolicyEntryContractSchemaVersion = 1;
constexpr int kHandoffAssessmentSchemaVersion = 1;
constexpr int kSourceExitStatusSchemaVersion = 2;
constexpr int kExternalInputSnapshotSchemaVersion = 1;
constexpr int kHrmFeatureSchemaVersion = 1;

enum class PreviewCapability {
  kUnsupported,
  kLocalSignatureOnly,
  kTransactional,
};

const char* preview_capability_name(PreviewCapability capability);

enum class HandoffDecision {
  kPending,
  kDirect,
  kRejectUnsafeSource,
  kRecover,
  kHold,
};

const char* handoff_decision_name(HandoffDecision decision);

struct SourceExitStatusV1 {
  int schema_version{kSourceExitStatusSchemaVersion};
  bool valid{false};
  uint64_t sequence{0};
  double source_timestamp_sec{std::numeric_limits<double>::quiet_NaN()};
  double received_timestamp_sec{std::numeric_limits<double>::quiet_NaN()};
  double freshness_sec{std::numeric_limits<double>::infinity()};
  std::string phase;
  std::string contact_mode;
  int contact_count{-1};
  bool airborne{false};
  bool rotation{false};
  bool fallen{false};
  double base_height_m{std::numeric_limits<double>::quiet_NaN()};
  double tilt_deg{std::numeric_limits<double>::quiet_NaN()};
  double angular_speed_rps{std::numeric_limits<double>::quiet_NaN()};
  std::string retry_policy{"retry_on_safe_window"};
};

struct SourceExitGateConfig {
  bool required{false};
  double maximum_freshness_sec{0.25};
  int minimum_contact_count{1};
  double minimum_base_height_m{0.55};
  double maximum_tilt_deg{35.0};
  double maximum_angular_speed_rps{3.0};
};

struct SourceExitGateResult {
  bool admissible{true};
  HandoffDecision decision{HandoffDecision::kDirect};
  std::string reason{"source exit gate disabled"};
};

struct ExternalInputSnapshotV1 {
  int schema_version{kExternalInputSnapshotSchemaVersion};
  bool valid{false};
  std::string source;
  uint64_t sequence{0};
  double source_timestamp_sec{std::numeric_limits<double>::quiet_NaN()};
  double capture_timestamp_sec{std::numeric_limits<double>::quiet_NaN()};
  double freshness_sec{std::numeric_limits<double>::infinity()};
  std::string content_hash;
};

struct HrmFeatureVectorV1 {
  int schema_version{kHrmFeatureSchemaVersion};
  std::vector<std::string> names;
  std::vector<float> values;
};

// Runtime-visible entry contract.  It deliberately describes only properties
// that can be checked at deployment time; it is not a claim that tensor shapes
// establish observation or action semantics.
struct PolicyEntryContractV1 {
  int schema_version{kPolicyEntryContractSchemaVersion};
  std::string policy;
  PreviewCapability preview_capability{PreviewCapability::kUnsupported};
  double activation_horizon_sec{2.0};
  double minimum_dwell_sec{0.0};
  std::vector<std::string> hard_preconditions;
  std::vector<std::string> entry_feature_names;
  std::vector<double> entry_center;
  std::vector<double> entry_scale;
  std::string contact_mode;
  std::string phase;
  std::vector<std::string> safe_exit_phases;
  bool reject_airborne_exit{true};
  bool reject_rotation_exit{true};
  std::string native_startup_mode{"preserve"};
  std::string unsafe_handoff_behavior{"reject"};
  std::string completion_event;
};

struct HandoffAssessmentV1 {
  int schema_version{kHandoffAssessmentSchemaVersion};
  std::string run_id;
  std::string trial_id;
  std::string pair_id;
  std::string source_policy;
  std::string target_policy;
  std::string group;
  uint64_t source_sequence{0};
  uint64_t target_sequence{0};
  HandoffDecision decision{HandoffDecision::kPending};
  std::string source_phase;
  bool source_exit_admissible{true};
  bool target_viability{false};
  std::string retry_policy;
  std::string reject_reason;
  bool hard_preconditions_met{false};
  bool authorized{false};
  double failure_probability{std::numeric_limits<double>::quiet_NaN()};
  double uncertainty{std::numeric_limits<double>::quiet_NaN()};
  double target_age_sec{std::numeric_limits<double>::infinity()};
  double max_position_jump_rad{std::numeric_limits<double>::quiet_NaN()};
  double max_velocity_jump{std::numeric_limits<double>::quiet_NaN()};
  double max_torque_jump{std::numeric_limits<double>::quiet_NaN()};
  std::string reason;
};

bool validate_policy_entry_contract(const PolicyEntryContractV1& contract,
                                    std::string& error_message);
SourceExitGateResult evaluate_source_exit_gate(const SourceExitStatusV1& status,
                                               const SourceExitGateConfig& config);

}  // namespace action_go::runtime
