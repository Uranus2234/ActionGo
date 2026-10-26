#pragma once

#include <string>
#include <vector>

namespace action_go::runtime {

inline constexpr int kUnitreeG1ProtocolMotorCount = 29;

inline constexpr int kBfmProtocolLatentDim = 256;
inline constexpr int kBfmProtocolStateDim = 64;
inline constexpr int kBfmProtocolHistoryDim = 372;
inline constexpr int kBfmProtocolHistorySteps = 4;
inline constexpr int kBfmProtocolObsDim =
    kBfmProtocolStateDim + kUnitreeG1ProtocolMotorCount + kBfmProtocolHistoryDim + kBfmProtocolLatentDim;

inline constexpr int kPbfmProtocolHeadingDim = 2;
inline constexpr int kPbfmProtocolVelocityDim = 3;
inline constexpr int kPbfmProtocolDepthFrames = 8;
inline constexpr int kPbfmProtocolDepthHeight = 18;
inline constexpr int kPbfmProtocolDepthWidth = 32;
inline constexpr int kPbfmProtocolDepthSourceHeight = 36;
inline constexpr int kPbfmProtocolDepthSourceWidth = 64;
inline constexpr int kPbfmProtocolDepthHistoryLength = 37;
inline constexpr int kPbfmProtocolObsDim = kBfmProtocolObsDim + kPbfmProtocolHeadingDim +
    kPbfmProtocolVelocityDim + kPbfmProtocolDepthFrames * kPbfmProtocolDepthHeight * kPbfmProtocolDepthWidth;

inline constexpr int kEmbeddedMimicProtocolObsDim = 154;
inline constexpr int kBeyondMimicProtocolRefBodyCount = 14;
inline constexpr int kBeyondMimicProtocolRefAnchorIndex = 7;

inline constexpr int kSuperMimicProtocolObsFrameDim = 99;
inline constexpr int kSuperMimicProtocolHistoryLen = 10;
inline constexpr int kSuperMimicProtocolGoalFrames = 10;
inline constexpr int kSuperMimicProtocolGoalDim = 730;
inline constexpr int kSuperMimicProtocolObsDim =
    kSuperMimicProtocolObsFrameDim * kSuperMimicProtocolHistoryLen + kSuperMimicProtocolGoalDim;

inline constexpr int kUnitreeRlMjlabProtocolBaseObsDim =
    3 + 3 + 3 + kUnitreeG1ProtocolMotorCount * 3;

inline constexpr int kTwistProtocolActionDim = 23;
inline constexpr int kTwistProtocolMimicDim = 33;
inline constexpr int kTwistProtocolWristDim = 6;
inline constexpr int kTwistProtocolPolicyMimicDim = 31;
inline constexpr int kTwistProtocolProprioDim = 74;
inline constexpr int kTwistProtocolObsFrameDim =
    kTwistProtocolPolicyMimicDim + kTwistProtocolProprioDim;
inline constexpr int kTwistProtocolHistoryLen = 10;
inline constexpr int kTwistProtocolObsDim =
    kTwistProtocolObsFrameDim * (1 + kTwistProtocolHistoryLen);

inline constexpr int kHumanoidGptProtocolObsDim = 136;
inline constexpr int kHumanoidGptProtocolQposDim = 36;
inline constexpr int kHumanoidGptProtocolQvelDim = 35;
inline constexpr int kHumanoidGptProtocolKeypointCount = 14;
inline constexpr int kHumanoidGptProtocolHmcpHeaderSize = 20;

inline constexpr int kSonicEncoderObsDim = 1762;
inline constexpr int kSonicV11EncoderObsDim = 1751;
inline constexpr int kSonicTokenDim = 64;
inline constexpr int kSonicDecoderObsDim = 994;
inline constexpr int kSonicHistoryLen = 10;
inline constexpr int kSonicReferenceFrames = 10;
inline constexpr int kSonicReferenceStep = 5;
inline constexpr int kSonicSmplJointCount = 24;
inline constexpr int kSonicSmplPoseCount = 21;
inline constexpr int kSonicSmplReferenceFrames = 10;
inline constexpr int kSonicZmqHeaderSize = 1280;
inline constexpr int kSonicPlannerContextFrames = 4;
inline constexpr int kSonicPlannerQposDim = 36;
inline constexpr int kSonicPlannerOutputFrames = 64;
inline constexpr int kSonicPlannerModeCount = 27;

inline constexpr int kInstinctProtocolObsDim = 5376;
inline constexpr int kInstinctProtocolProprioDim = 768;
inline constexpr int kInstinctProtocolDepthLatentDim = 128;
inline constexpr int kInstinctProtocolActorInputDim = 896;
inline constexpr int kPhpProtocolObsDim = 140;
inline constexpr int kPhpProtocolCommandDim = 15;
inline constexpr int kPhpProtocolDepthLatentDim = 32;

inline constexpr int kSteadyTrayProtocolObsDim = 480;
inline constexpr int kSteadyTrayProtocolFrameDim = 96;
inline constexpr int kSteadyTrayProtocolPolicyFrames = 5;
inline constexpr int kSteadyTrayProtocolEncoderObsDim = 103;
inline constexpr int kSteadyTrayProtocolEncoderHistoryLen = 32;

inline constexpr int kOmniXtremeProtocolObsDim = 1687;
inline constexpr int kOmniXtremeProtocolRealObsDim = 90;
inline constexpr int kOmniXtremeProtocolCommandObsDim = 64;
inline constexpr int kOmniXtremeProtocolHistoryDim = 1350;
inline constexpr int kOmniXtremeProtocolResidualObsDim = 183;

struct PolicySessionProtocolConstraint {
  const char* name;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
};

struct PolicyNamedDim {
  const char* name;
  int value;
};

struct PolicyYamlDimRule {
  const char* path;
  int expected;
  const char* label;
};

struct PolicyProtocolConstraint {
  const char* variant;
  const char* robot_model;
  bool unitree_g1_specific;
  bool optional_build;
  const char* optional_build_note;

  int motor_count;
  int obs_dim;
  int action_dim;

  std::vector<const char*> model_input_names;
  std::vector<const char*> model_output_names;
  std::vector<PolicySessionProtocolConstraint> sessions;

  int latent_dim;
  int history_steps;
  int base_obs_dim;
  std::vector<int> allowed_obs_dims;

  int ref_body_count;
  int ref_anchor_index;

  int twist_mimic_dim;
  int twist_wrist_dim;
  int twist_policy_mimic_dim;
  int twist_proprio_dim;
  int twist_obs_frame_dim;
  int twist_history_len;
  int twist_input_dim;

  int humanoid_gpt_qpos_dim;
  int humanoid_gpt_qvel_dim;
  int humanoid_gpt_keypoint_count;
  int humanoid_gpt_hmcp_header_size;
};

const PolicyProtocolConstraint* protocol_constraint_for_variant(const std::string& variant_name);
std::vector<std::string> protocol_names(const std::vector<const char*>& names);
std::vector<PolicyNamedDim> protocol_named_dims_for_variant(const std::string& variant_name);
std::vector<PolicyYamlDimRule> protocol_yaml_dim_rules_for_variant(const std::string& variant_name);
std::string format_protocol_constraint_summary(const PolicyProtocolConstraint& constraint);

}  // namespace action_go::runtime
