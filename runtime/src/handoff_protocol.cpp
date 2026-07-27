#include "action_go/runtime/handoff_protocol.h"

#include <algorithm>
#include <cmath>

namespace action_go::runtime {

const char* preview_capability_name(PreviewCapability capability) {
  switch (capability) {
    case PreviewCapability::kUnsupported: return "unsupported";
    case PreviewCapability::kLocalSignatureOnly: return "local_signature_only";
    case PreviewCapability::kTransactional: return "transactional";
  }
  return "unsupported";
}

const char* handoff_decision_name(HandoffDecision decision) {
  switch (decision) {
    case HandoffDecision::kPending: return "pending";
    case HandoffDecision::kDirect: return "direct";
    case HandoffDecision::kRejectUnsafeSource: return "reject_unsafe_source";
    case HandoffDecision::kRecover: return "recover";
    case HandoffDecision::kHold: return "hold";
  }
  return "pending";
}

bool validate_policy_entry_contract(const PolicyEntryContractV1& contract,
                                    std::string& error_message) {
  if (contract.schema_version != kPolicyEntryContractSchemaVersion) {
    error_message = "unsupported PolicyEntryContract schema_version=" +
                    std::to_string(contract.schema_version);
    return false;
  }
  if (contract.policy.empty()) {
    error_message = "PolicyEntryContract policy is empty";
    return false;
  }
  if (!std::isfinite(contract.activation_horizon_sec) ||
      contract.activation_horizon_sec <= 0.0) {
    error_message = "PolicyEntryContract activation_horizon_sec must be positive";
    return false;
  }
  if (!std::isfinite(contract.minimum_dwell_sec) || contract.minimum_dwell_sec < 0.0) {
    error_message = "PolicyEntryContract minimum_dwell_sec must be non-negative";
    return false;
  }
  if (contract.native_startup_mode != "preserve" &&
      contract.native_startup_mode != "raw_direct") {
    error_message = "PolicyEntryContract native_startup_mode must be preserve or raw_direct";
    return false;
  }
  if (contract.unsafe_handoff_behavior != "reject" &&
      contract.unsafe_handoff_behavior != "recover") {
    error_message = "PolicyEntryContract unsafe_handoff_behavior must be reject or recover";
    return false;
  }
  if (contract.entry_center.size() != contract.entry_scale.size()) {
    error_message = "PolicyEntryContract entry_center/entry_scale size mismatch";
    return false;
  }
  if (!contract.entry_feature_names.empty() &&
      contract.entry_feature_names.size() != contract.entry_center.size()) {
    error_message = "PolicyEntryContract feature names do not match entry vector size";
    return false;
  }
  for (size_t index = 0; index < contract.entry_center.size(); ++index) {
    if (!std::isfinite(contract.entry_center[index])) {
      error_message = "PolicyEntryContract entry_center contains non-finite value";
      return false;
    }
    if (!std::isfinite(contract.entry_scale[index]) || contract.entry_scale[index] < 0.0) {
      error_message = "PolicyEntryContract entry_scale must be finite and non-negative";
      return false;
    }
  }
  error_message.clear();
  return true;
}

SourceExitGateResult evaluate_source_exit_gate(const SourceExitStatusV1& status,
                                               const SourceExitGateConfig& config) {
  if (!config.required) return {};

  SourceExitGateResult result;
  result.admissible = false;
  result.decision = HandoffDecision::kRejectUnsafeSource;
  if (status.schema_version != kSourceExitStatusSchemaVersion || !status.valid) {
    result.reason = "source exit status unavailable";
    return result;
  }
  if (status.sequence == 0) {
    result.reason = "source exit status has no sequence";
    return result;
  }
  if (!std::isfinite(status.freshness_sec) || status.freshness_sec < 0.0 ||
      status.freshness_sec > config.maximum_freshness_sec) {
    result.reason = "source exit status is stale";
    return result;
  }
  if (status.fallen) {
    result.decision = HandoffDecision::kRecover;
    result.reason = "source state requires dedicated recovery";
    return result;
  }
  if (status.airborne) {
    result.reason = "source phase is airborne";
    return result;
  }
  if (status.rotation) {
    result.reason = "source phase is unsafe rotation";
    return result;
  }
  if (status.contact_count < config.minimum_contact_count) {
    result.reason = "source has no reliable support contact";
    return result;
  }
  if (!std::isfinite(status.base_height_m) ||
      status.base_height_m < config.minimum_base_height_m) {
    result.reason = "source base height is below the recoverable upright domain";
    return result;
  }
  if (!std::isfinite(status.tilt_deg) ||
      std::abs(status.tilt_deg) > config.maximum_tilt_deg) {
    result.reason = "source torso tilt exceeds the exit limit";
    return result;
  }
  if (!std::isfinite(status.angular_speed_rps) ||
      std::abs(status.angular_speed_rps) > config.maximum_angular_speed_rps) {
    result.reason = "source angular speed exceeds the exit limit";
    return result;
  }
  result.admissible = true;
  result.decision = HandoffDecision::kDirect;
  result.reason = "source exit admissible";
  return result;
}

}  // namespace action_go::runtime
