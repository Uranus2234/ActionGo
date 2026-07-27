#include "action_go/runtime/policy_protocol.h"

#include <sstream>

namespace action_go::runtime {
namespace {

const std::vector<PolicyProtocolConstraint>& constraints() {
  static const std::vector<PolicyProtocolConstraint> kConstraints{
      {"bfm_zero",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kBfmProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {"actor_obs"},
       {"action"},
       {},
       kBfmProtocolLatentDim,
       kBfmProtocolHistorySteps,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"unitree_rl_mjlab",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kUnitreeRlMjlabProtocolBaseObsDim + 2,
       kUnitreeG1ProtocolMotorCount,
       {"obs"},
       {"actions"},
       {},
       0,
       0,
       kUnitreeRlMjlabProtocolBaseObsDim,
       {kUnitreeRlMjlabProtocolBaseObsDim,
        kUnitreeRlMjlabProtocolBaseObsDim + 1,
        kUnitreeRlMjlabProtocolBaseObsDim + 2},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"beyondmimic",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kEmbeddedMimicProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {"obs", "time_step"},
       {"actions", "joint_pos", "joint_vel", "body_quat_w"},
       {},
       0,
       0,
       0,
       {},
       kBeyondMimicProtocolRefBodyCount,
       kBeyondMimicProtocolRefAnchorIndex,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"holosoma",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kEmbeddedMimicProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {"obs", "time_step"},
       {"actions", "joint_pos", "joint_vel", "ref_quat_xyzw"},
       {},
       0,
       0,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"protomotions",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       342,
       kUnitreeG1ProtocolMotorCount,
       {"current_anchor_rot",
        "current_dof_pos",
        "current_dof_vel",
        "current_root_local_ang_vel",
        "historical_processed_actions",
        "mimic_future_anchor_rot",
        "mimic_future_dof_pos",
        "mimic_future_dof_vel"},
       {"actions", "joint_pos_targets", "stiffness_targets", "damping_targets"},
       {},
       0,
       0,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"twist",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kTwistProtocolObsDim,
       kTwistProtocolActionDim,
       {"obs"},
       {"action"},
       {},
       0,
       0,
       0,
       {},
       0,
       0,
       kTwistProtocolMimicDim,
       kTwistProtocolWristDim,
       kTwistProtocolPolicyMimicDim,
       kTwistProtocolProprioDim,
       kTwistProtocolObsFrameDim,
       kTwistProtocolHistoryLen,
       kTwistProtocolObsDim,
       0,
       0,
       0,
       0},
      {"humanoid_gpt",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kHumanoidGptProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {"obs"},
       {"continuous_actions"},
       {},
       0,
       0,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       kHumanoidGptProtocolQposDim,
       kHumanoidGptProtocolQvelDim,
       kHumanoidGptProtocolKeypointCount,
       kHumanoidGptProtocolHmcpHeaderSize},
      {"sonic",
       "Unitree G1 29DoF",
       true,
       false,
       "SONIC uses encoder and decoder ONNX sessions; TensorRT is selected by runtime configuration",
       kUnitreeG1ProtocolMotorCount,
       kSonicDecoderObsDim,
       kUnitreeG1ProtocolMotorCount,
       {},
       {},
       {{"encoder", {"obs_dict"}, {"encoded_tokens"}},
        {"decoder", {"obs_dict"}, {"action"}}},
       kSonicTokenDim,
       kSonicHistoryLen,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"instinct_parkour",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kInstinctProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {},
       {},
       {{"depth_encoder", {"input"}, {"output"}},
        {"actor", {"input"}, {"output"}}},
       0,
       0,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"instinct",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kInstinctProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {},
       {},
       {{"depth_encoder", {"input"}, {"output"}},
        {"actor", {"input"}, {"output"}}},
       0,
       0,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"steadytray",
       "Unitree G1 29DoF",
       true,
       false,
       "",
       kUnitreeG1ProtocolMotorCount,
       kSteadyTrayProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {"student_encoder_obs", "policy_obs"},
       {"actions"},
       {},
       0,
       0,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
      {"omnixtreme",
       "Unitree G1 29DoF",
       true,
       false,
       "generic_composite_policy_player builds by default; TensorRT is selected by runtime provider configuration",
       kUnitreeG1ProtocolMotorCount,
       kOmniXtremeProtocolObsDim,
       kUnitreeG1ProtocolMotorCount,
       {},
       {},
       {{"base", {"real_obs", "command_obs", "real_historical_obs_raw", "initial_noise"}, {"action"}},
        {"residual", {"obs"}, {"actions"}},
        {"fk", {"joint_angles", "base_pos", "base_quat"}, {"pos", "rot"}}},
       0,
       0,
       0,
       {},
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0},
  };
  return kConstraints;
}

}  // namespace

const PolicyProtocolConstraint* protocol_constraint_for_variant(const std::string& variant_name) {
  for (const PolicyProtocolConstraint& constraint : constraints()) {
    if (variant_name == constraint.variant) return &constraint;
  }
  return nullptr;
}

std::vector<std::string> protocol_names(const std::vector<const char*>& names) {
  std::vector<std::string> out;
  out.reserve(names.size());
  for (const char* name : names) out.emplace_back(name ? name : "");
  return out;
}

std::vector<PolicyNamedDim> protocol_named_dims_for_variant(const std::string& variant_name) {
  if (variant_name == "bfm_zero") {
    return {{"latent_dim", kBfmProtocolLatentDim},
            {"history_steps", kBfmProtocolHistorySteps}};
  }
  if (variant_name == "unitree_rl_mjlab") {
    return {{"base_obs_dim", kUnitreeRlMjlabProtocolBaseObsDim}};
  }
  if (variant_name == "beyondmimic") {
    return {{"ref_body_count", kBeyondMimicProtocolRefBodyCount},
            {"anchor_body_index", kBeyondMimicProtocolRefAnchorIndex}};
  }
  if (variant_name == "twist") {
    return {{"mimic_dim", kTwistProtocolMimicDim},
            {"wrist_dim", kTwistProtocolWristDim},
            {"policy_mimic_dim", kTwistProtocolPolicyMimicDim},
            {"proprio_dim", kTwistProtocolProprioDim},
            {"obs_frame_dim", kTwistProtocolObsFrameDim},
            {"history_len", kTwistProtocolHistoryLen},
            {"input_dim", kTwistProtocolObsDim}};
  }
  if (variant_name == "humanoid_gpt") {
    return {{"qpos_dim", kHumanoidGptProtocolQposDim},
            {"qvel_dim", kHumanoidGptProtocolQvelDim},
            {"keypoint_count", kHumanoidGptProtocolKeypointCount},
            {"hmcp_header_size", kHumanoidGptProtocolHmcpHeaderSize}};
  }
  if (variant_name == "sonic") {
    return {{"encoder_obs_dim", kSonicEncoderObsDim},
            {"token_dim", kSonicTokenDim},
            {"history_len", kSonicHistoryLen},
            {"smpl_joint_count", kSonicSmplJointCount},
            {"smpl_pose_count", kSonicSmplPoseCount},
            {"smpl_reference_frames", kSonicSmplReferenceFrames},
            {"reference_frames", kSonicReferenceFrames},
            {"reference_step", kSonicReferenceStep},
            {"zmq_header_size", kSonicZmqHeaderSize}};
  }
  if (variant_name == "instinct_parkour" || variant_name == "instinct") {
    return {{"proprio_dim", kInstinctProtocolProprioDim},
            {"depth_latent_dim", kInstinctProtocolDepthLatentDim},
            {"actor_input_dim", kInstinctProtocolActorInputDim}};
  }
  if (variant_name == "steadytray") {
    return {{"frame_dim", kSteadyTrayProtocolFrameDim},
            {"policy_frames", kSteadyTrayProtocolPolicyFrames},
            {"encoder_obs_dim", kSteadyTrayProtocolEncoderObsDim},
            {"encoder_history_len", kSteadyTrayProtocolEncoderHistoryLen}};
  }
  if (variant_name == "omnixtreme") {
    return {{"real_obs_dim", kOmniXtremeProtocolRealObsDim},
            {"command_obs_dim", kOmniXtremeProtocolCommandObsDim},
            {"history_dim", kOmniXtremeProtocolHistoryDim},
            {"residual_obs_dim", kOmniXtremeProtocolResidualObsDim}};
  }
  return {};
}

std::vector<PolicyYamlDimRule> protocol_yaml_dim_rules_for_variant(const std::string& variant_name) {
  if (variant_name == "bfm_zero") {
    return {{"observation.latent_dim", kBfmProtocolLatentDim, "BFM-Zero latent dim"},
            {"observation.history_steps", kBfmProtocolHistorySteps, "BFM-Zero history steps"}};
  }
  if (variant_name == "unitree_rl_mjlab") {
    return {{"observation.base_obs_dim", kUnitreeRlMjlabProtocolBaseObsDim, "unitree base obs dim"}};
  }
  if (variant_name == "beyondmimic") {
    return {{"mimic_time.ref_body_count", kBeyondMimicProtocolRefBodyCount, "BeyondMimic ref body count"},
            {"mimic_time.anchor_body_index", kBeyondMimicProtocolRefAnchorIndex, "BeyondMimic anchor body index"}};
  }
  if (variant_name == "twist") {
    return {{"motion_command.mimic_dim", kTwistProtocolMimicDim, "TWIST motion_command.mimic_dim"},
            {"motion_command.wrist_dim", kTwistProtocolWristDim, "TWIST motion_command.wrist_dim"},
            {"observation.policy_mimic_dim", kTwistProtocolPolicyMimicDim, "TWIST policy mimic dim"},
            {"observation.proprio_dim", kTwistProtocolProprioDim, "TWIST proprio dim"},
            {"observation.obs_frame_dim", kTwistProtocolObsFrameDim, "TWIST obs frame dim"},
            {"observation.history_len", kTwistProtocolHistoryLen, "TWIST history len"},
            {"observation.input_dim", kTwistProtocolObsDim, "TWIST input dim"}};
  }
  if (variant_name == "humanoid_gpt") {
    return {{"reference.qpos_dim", kHumanoidGptProtocolQposDim, "Humanoid-GPT qpos dim"},
            {"reference.qvel_dim", kHumanoidGptProtocolQvelDim, "Humanoid-GPT qvel dim"},
            {"reference.keypoint_count", kHumanoidGptProtocolKeypointCount, "Humanoid-GPT keypoint count"},
            {"reference.hmcp_header_size", kHumanoidGptProtocolHmcpHeaderSize, "Humanoid-GPT HMCP header size"}};
  }
  if (variant_name == "instinct_parkour" || variant_name == "instinct") {
    return {{"observation.proprio_dim", kInstinctProtocolProprioDim, "Instinct proprio dim"},
            {"observation.depth_latent_dim", kInstinctProtocolDepthLatentDim, "Instinct depth latent dim"},
            {"observation.actor_input_dim", kInstinctProtocolActorInputDim, "Instinct actor input dim"}};
  }
  if (variant_name == "steadytray") {
    return {{"observation.frame_dim", kSteadyTrayProtocolFrameDim, "SteadyTray frame dim"},
            {"observation.policy_frames", kSteadyTrayProtocolPolicyFrames, "SteadyTray policy frames"},
            {"observation.encoder_obs_dim", kSteadyTrayProtocolEncoderObsDim, "SteadyTray encoder obs dim"},
            {"observation.encoder_history_len", kSteadyTrayProtocolEncoderHistoryLen, "SteadyTray encoder history len"}};
  }
  if (variant_name == "omnixtreme") {
    return {{"observation.real_obs_dim", kOmniXtremeProtocolRealObsDim, "OmniXtreme real obs dim"},
            {"observation.command_obs_dim", kOmniXtremeProtocolCommandObsDim, "OmniXtreme command obs dim"},
            {"observation.history_dim", kOmniXtremeProtocolHistoryDim, "OmniXtreme history dim"},
            {"observation.residual_obs_dim", kOmniXtremeProtocolResidualObsDim, "OmniXtreme residual obs dim"}};
  }
  return {};
}

std::string format_protocol_constraint_summary(const PolicyProtocolConstraint& constraint) {
  std::ostringstream out;
  out << "robot=" << constraint.robot_model
      << " g1_specific=" << (constraint.unitree_g1_specific ? "true" : "false")
      << " motor_count=" << constraint.motor_count
      << " obs_dim=" << constraint.obs_dim
      << " action_dim=" << constraint.action_dim;
  if (constraint.twist_mimic_dim > 0) {
    out << " mimic_dim=" << constraint.twist_mimic_dim
        << " wrist_dim=" << constraint.twist_wrist_dim
        << " history_len=" << constraint.twist_history_len;
  }
  if (constraint.humanoid_gpt_qpos_dim > 0) {
    out << " qpos_dim=" << constraint.humanoid_gpt_qpos_dim
        << " qvel_dim=" << constraint.humanoid_gpt_qvel_dim
        << " keypoints=" << constraint.humanoid_gpt_keypoint_count;
  }
  if (constraint.optional_build) {
    out << " optional_build=true";
  }
  const auto named_dims = protocol_named_dims_for_variant(constraint.variant);
  if (!named_dims.empty()) {
    out << " dims={";
    for (size_t i = 0; i < named_dims.size(); ++i) {
      if (i) out << ",";
      out << named_dims[i].name << ":" << named_dims[i].value;
    }
    out << "}";
  }
  return out.str();
}

}  // namespace action_go::runtime
