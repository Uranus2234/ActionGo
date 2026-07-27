#pragma once

#include "action_go/runtime/handoff_protocol.h"

#include <string>
#include <vector>

namespace action_go::runtime {

enum class PolicyTemplateId {
  kSingleActorState,
  kEmbeddedMimicTime,
  kMotionConditionedTracking,
  kDepthConditionedActor,
  kObjectConditionedActor,
  kCompositeMotionPolicy,
};

enum class GenericPlayerKind {
  kPolicy,
  kMotionTracking,
  kSensor,
  kComposite,
};

enum class ObservationSourceKind {
  kRobotState,
  kRobotStateAndTime,
  kMotionConditioned,
  kDepthConditioned,
  kObjectConditioned,
  kCompositeMotion,
};

enum class ActionDecoderKind {
  kScaledAction,
  kEmbeddedReferenceAction,
  kTrackingTarget,
  kSensorActor,
  kCompositePipeline,
};

enum class ExternalSourceKind {
  kNone,
  kTimeStep,
  kMotionCacheOrRealtime,
  kDepthSensor,
  kObjectPose,
  kMotionTrajectory,
};

struct PolicyVariantContract {
  const char* name;
  bool core_paper_coverage;
  PreviewCapability preview_capability;
  bool freezes_external_input;
};

struct PolicyContract {
  PolicyTemplateId id;
  const char* template_name;
  const char* category_name;
  GenericPlayerKind player_kind;
  ObservationSourceKind observation_source;
  ActionDecoderKind action_decoder;
  ExternalSourceKind external_source;
  bool core_paper_coverage;
  std::vector<const char*> aliases;
  std::vector<PolicyVariantContract> variants;
};

const PolicyContract* contract_for_template(const std::string& template_name);
const PolicyContract* contract_for_variant(const std::string& variant_name);
const PolicyVariantContract* variant_contract(const std::string& variant_name);

bool template_allowed_for_player(const std::string& template_name, GenericPlayerKind player_kind);
bool variant_allowed_for_player(const std::string& variant_name, GenericPlayerKind player_kind);
bool variant_allowed_for_template(const std::string& template_name, const std::string& variant_name);

std::string player_binary_name(GenericPlayerKind player_kind);
std::string format_supported_templates(GenericPlayerKind player_kind);
std::string format_supported_variants(GenericPlayerKind player_kind);
std::string format_supported_variants_for_template(const std::string& template_name);

}  // namespace action_go::runtime
