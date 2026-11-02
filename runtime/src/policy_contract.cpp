#include "action_go/runtime/policy_contract.h"

#include <algorithm>
#include <sstream>

namespace action_go::runtime {
namespace {

const std::vector<PolicyContract>& contracts() {
  static const std::vector<PolicyContract> kContracts{
      {PolicyTemplateId::kSingleActorState,
       "single_actor_state",
       "State/Prompt-Conditioned Actor Policy",
       GenericPlayerKind::kPolicy,
       ObservationSourceKind::kRobotState,
       ActionDecoderKind::kScaledAction,
       ExternalSourceKind::kNone,
       true,
       {},
       {{"bfm_zero", true, PreviewCapability::kTransactional, false},
        {"unitree_rl_mjlab", false, PreviewCapability::kUnsupported, false}}},
      {PolicyTemplateId::kEmbeddedMimicTime,
       "embedded_mimic_time",
       "State/Prompt-Conditioned Actor Policy",
       GenericPlayerKind::kPolicy,
       ObservationSourceKind::kRobotStateAndTime,
       ActionDecoderKind::kEmbeddedReferenceAction,
       ExternalSourceKind::kTimeStep,
       true,
       {},
       {{"beyondmimic", true, PreviewCapability::kTransactional, false},
        {"holosoma", false, PreviewCapability::kUnsupported, false}}},
      {PolicyTemplateId::kMotionConditionedTracking,
       "motion_conditioned_tracking",
       "Motion-Conditioned Tracking Policy",
       GenericPlayerKind::kMotionTracking,
       ObservationSourceKind::kMotionConditioned,
       ActionDecoderKind::kTrackingTarget,
       ExternalSourceKind::kMotionCacheOrRealtime,
       true,
       {"external_reference_tracking"},
       {{"protomotions", true, PreviewCapability::kTransactional, false},
        {"twist", true, PreviewCapability::kTransactional, true},
        {"humanoid_gpt", false, PreviewCapability::kUnsupported, true},
        {"sonic", false, PreviewCapability::kUnsupported, true},
        {"supermimic", false, PreviewCapability::kUnsupported, false},
        {"uni_supermimic", false, PreviewCapability::kUnsupported, false},
        {"allmimic", false, PreviewCapability::kUnsupported, false},
        {"mjlab_combat", false, PreviewCapability::kUnsupported, false}}},
      {PolicyTemplateId::kDepthConditionedActor,
       "depth_conditioned_actor",
       "Sensor-Conditioned Actor Policy",
       GenericPlayerKind::kSensor,
       ObservationSourceKind::kDepthConditioned,
       ActionDecoderKind::kSensorActor,
       ExternalSourceKind::kDepthSensor,
       true,
       {},
       {{"instinct_parkour", true, PreviewCapability::kTransactional, true},
        {"instinct", true, PreviewCapability::kTransactional, true},
        {"php", false, PreviewCapability::kTransactional, true},
        {"pbfm", false, PreviewCapability::kUnsupported, true}}},
      {PolicyTemplateId::kObjectConditionedActor,
       "object_conditioned_actor",
       "Sensor-Conditioned Actor Policy",
       GenericPlayerKind::kSensor,
       ObservationSourceKind::kObjectConditioned,
       ActionDecoderKind::kSensorActor,
       ExternalSourceKind::kObjectPose,
       true,
       {},
       {{"steadytray", true, PreviewCapability::kTransactional, true}}},
      {PolicyTemplateId::kCompositeMotionPolicy,
       "composite_motion_policy",
       "Composite Motion Pipeline Policy",
       GenericPlayerKind::kComposite,
       ObservationSourceKind::kCompositeMotion,
       ActionDecoderKind::kCompositePipeline,
       ExternalSourceKind::kMotionTrajectory,
       true,
       {},
       {{"omnixtreme", true, PreviewCapability::kTransactional, true}}},
  };
  return kContracts;
}

bool string_in_list(const std::string& value, const std::vector<const char*>& list) {
  return std::any_of(list.begin(), list.end(), [&](const char* item) {
    return value == item;
  });
}

bool has_variant(const PolicyContract& contract, const std::string& variant_name) {
  return std::any_of(contract.variants.begin(), contract.variants.end(), [&](const PolicyVariantContract& variant) {
    return variant_name == variant.name;
  });
}

std::string join_unique(const std::vector<std::string>& values) {
  std::vector<std::string> unique;
  for (const std::string& value : values) {
    if (std::find(unique.begin(), unique.end(), value) == unique.end()) unique.push_back(value);
  }
  std::ostringstream out;
  for (size_t i = 0; i < unique.size(); ++i) {
    if (i) out << ", ";
    out << unique[i];
  }
  return out.str();
}

}  // namespace

const PolicyContract* contract_for_template(const std::string& template_name) {
  for (const PolicyContract& contract : contracts()) {
    if (template_name == contract.template_name || string_in_list(template_name, contract.aliases)) {
      return &contract;
    }
  }
  return nullptr;
}

const PolicyContract* contract_for_variant(const std::string& variant_name) {
  for (const PolicyContract& contract : contracts()) {
    if (has_variant(contract, variant_name)) return &contract;
  }
  return nullptr;
}

const PolicyVariantContract* variant_contract(const std::string& variant_name) {
  for (const PolicyContract& contract : contracts()) {
    for (const PolicyVariantContract& variant : contract.variants) {
      if (variant_name == variant.name) return &variant;
    }
  }
  return nullptr;
}

bool template_allowed_for_player(const std::string& template_name, GenericPlayerKind player_kind) {
  const PolicyContract* contract = contract_for_template(template_name);
  return contract && contract->player_kind == player_kind;
}

bool variant_allowed_for_player(const std::string& variant_name, GenericPlayerKind player_kind) {
  const PolicyContract* contract = contract_for_variant(variant_name);
  return contract && contract->player_kind == player_kind;
}

bool variant_allowed_for_template(const std::string& template_name, const std::string& variant_name) {
  const PolicyContract* contract = contract_for_template(template_name);
  return contract && has_variant(*contract, variant_name);
}

std::string player_binary_name(GenericPlayerKind player_kind) {
  switch (player_kind) {
    case GenericPlayerKind::kPolicy:
      return "generic_policy_player";
    case GenericPlayerKind::kMotionTracking:
      return "generic_motion_tracking_player";
    case GenericPlayerKind::kSensor:
      return "generic_sensor_policy_player";
    case GenericPlayerKind::kComposite:
      return "generic_composite_policy_player";
  }
  return "generic_policy_player";
}

std::string format_supported_templates(GenericPlayerKind player_kind) {
  std::vector<std::string> names;
  for (const PolicyContract& contract : contracts()) {
    if (contract.player_kind == player_kind) names.emplace_back(contract.template_name);
  }
  return join_unique(names);
}

std::string format_supported_variants(GenericPlayerKind player_kind) {
  std::vector<std::string> names;
  for (const PolicyContract& contract : contracts()) {
    if (contract.player_kind != player_kind) continue;
    for (const PolicyVariantContract& variant : contract.variants) names.emplace_back(variant.name);
  }
  return join_unique(names);
}

std::string format_supported_variants_for_template(const std::string& template_name) {
  const PolicyContract* contract = contract_for_template(template_name);
  if (!contract) return {};
  std::vector<std::string> names;
  for (const PolicyVariantContract& variant : contract->variants) names.emplace_back(variant.name);
  return join_unique(names);
}

}  // namespace action_go::runtime
