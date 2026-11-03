#include "generic_policy_runtime_internal.h"

namespace action_go::runtime::generic_policy_runtime {

void GenericPolicyPlayer::load_spec_protomotions(const YAML::Node& y) {
    const YAML::Node reference = y["reference"];
    proto_anchor_source_ = scalar_or<std::string>(reference, "anchor_source", "fk");
    proto_startup_interp_sec_ = scalar_or<double>(reference, "startup_interp_sec", 0.0);
    proto_loop_motion_ = scalar_or<bool>(reference, "loop", false);
    if (proto_anchor_source_ != "fk" && proto_anchor_source_ != "imu") {
      throw std::runtime_error("reference.anchor_source must be fk or imu");
    }
  }

void GenericPolicyPlayer::load_spec_twist(const YAML::Node& y) {
    const YAML::Node runtime = y["runtime"];
    const YAML::Node motion_command = y["motion_command"];
    twist_policy_actions_ = configured_action_dim_;
    twist_policy_mimic_dim_ = yamlc::required_scalar<int>(y, "observation.policy_mimic_dim");
    twist_proprio_dim_ = yamlc::required_scalar<int>(y, "observation.proprio_dim");
    twist_history_len_ = yamlc::required_scalar<int>(y, "observation.history_len");
    twist_obs_frame_dim_ = yamlc::required_scalar<int>(y, "observation.obs_frame_dim");
    twist_input_dim_ = yamlc::required_scalar<int>(y, "observation.input_dim");
    twist_obs_dim_ = configured_obs_dim_;
    twist_full_mimic_dim_ = yamlc::required_scalar<int>(y, "motion_command.mimic_dim");
    twist_wrist_dim_ = yamlc::required_scalar<int>(y, "motion_command.wrist_dim");
    if (twist_full_mimic_dim_ != protocol_->twist_mimic_dim) {
      throw std::runtime_error("TWIST protocol constraint: motion_command.mimic_dim must be " +
                               std::to_string(protocol_->twist_mimic_dim));
    }
    if (twist_wrist_dim_ != protocol_->twist_wrist_dim) {
      throw std::runtime_error("TWIST protocol constraint: motion_command.wrist_dim must be " +
                               std::to_string(protocol_->twist_wrist_dim));
    }
    if (twist_policy_actions_ != protocol_->action_dim ||
        twist_policy_mimic_dim_ != protocol_->twist_policy_mimic_dim ||
        twist_proprio_dim_ != protocol_->twist_proprio_dim ||
        twist_history_len_ != protocol_->twist_history_len ||
        twist_obs_frame_dim_ != protocol_->twist_obs_frame_dim ||
        twist_input_dim_ != protocol_->twist_input_dim) {
      throw std::runtime_error("TWIST observation/action dimensions do not match protocol constraint");
    }
    if (twist_obs_frame_dim_ != twist_policy_mimic_dim_ + twist_proprio_dim_) {
      throw std::runtime_error("TWIST observation.obs_frame_dim must equal policy_mimic_dim + proprio_dim");
    }
    if (twist_input_dim_ != twist_obs_frame_dim_ * (1 + twist_history_len_) ||
        twist_obs_dim_ != twist_input_dim_) {
      throw std::runtime_error("TWIST observation.obs_dim/input_dim/history_len are inconsistent");
    }
    twist_input_source_ = lower(args_.input_override.empty()
                                    ? scalar_or<std::string>(motion_command, "source", scalar_or<std::string>(runtime, "input", "offline-pkl"))
                                    : args_.input_override);
    twist_realtime_transport_ = lower(args_.realtime_transport_override.empty()
                                          ? scalar_or<std::string>(motion_command, "realtime_transport", "udp")
                                          : args_.realtime_transport_override);
    if (twist_input_source_ == "udp" || twist_input_source_ == "realtime-udp") {
      twist_input_source_ = "realtime";
      twist_realtime_transport_ = "udp";
    } else if (twist_input_source_ == "zmq" || twist_input_source_ == "realtime-zmq") {
      twist_input_source_ = "realtime";
      twist_realtime_transport_ = "zmq";
    } else if (twist_input_source_ == "redis" || twist_input_source_ == "realtime-redis") {
      twist_input_source_ = "realtime";
      twist_realtime_transport_ = "redis";
    }
    std::string raw_motion_pkl = scalar_or<std::string>(motion_command, "pkl", scalar_or<std::string>(motion_command, "motion_pkl", ""));
    if (!args_.motion_pkl_override.empty()) raw_motion_pkl = args_.motion_pkl_override;
    else if (!args_.trajectory_override.empty()) raw_motion_pkl = args_.trajectory_override;
    if (!raw_motion_pkl.empty()) twist_motion_pkl_path_ = resolve_relative(spec_path_, raw_motion_pkl);
    std::string raw_motion_cache = scalar_or<std::string>(motion_command, "cache", scalar_or<std::string>(motion_command, "motion_cache", ""));
    if (!args_.motion_cache_override.empty()) raw_motion_cache = args_.motion_cache_override;
    if (!raw_motion_cache.empty()) twist_motion_cache_path_ = resolve_relative(spec_path_, raw_motion_cache);
    twist_root_ = args_.twist_root_override.empty() ? scalar_or<std::string>(motion_command, "twist_root", "") : args_.twist_root_override;
    twist_prepare_script_ = args_.prepare_script_override.empty() ? scalar_or<std::string>(motion_command, "prepare_script", "") : args_.prepare_script_override;
    twist_steps_ = args_.steps_override.empty() ? scalar_or<std::string>(motion_command, "steps", "1") : args_.steps_override;
    twist_rebuild_motion_cache_ = scalar_or<bool>(motion_command, "rebuild_cache", false) || args_.rebuild_motion_cache;
    twist_loop_motion_ = scalar_or<bool>(motion_command, "loop", false) || args_.loop;
    twist_require_realtime_ = scalar_or<bool>(motion_command, "require_realtime", false) || args_.require_realtime;
    const bool twist_stale_to_default = scalar_or<bool>(motion_command, "stale_to_default", false) || args_.stale_to_default;
    twist_udp_bind_ = args_.udp_bind_override.empty() ? scalar_or<std::string>(motion_command, "udp_bind", "127.0.0.1") : args_.udp_bind_override;
    twist_udp_port_ = args_.udp_port > 0 ? args_.udp_port : scalar_or<int>(motion_command, "udp_port", 15110);
    twist_zmq_endpoint_ = args_.zmq_endpoint_override.empty() ? scalar_or<std::string>(motion_command, "zmq_endpoint", "tcp://127.0.0.1:15111") : args_.zmq_endpoint_override;
    twist_zmq_topic_ = args_.zmq_topic_override.empty() ? scalar_or<std::string>(motion_command, "zmq_topic", "") : args_.zmq_topic_override;
    twist_zmq_bind_ = args_.zmq_bind_set ? args_.zmq_bind : scalar_or<bool>(motion_command, "zmq_bind", true);
    twist_redis_host_ = args_.redis_host_override.empty() ? scalar_or<std::string>(motion_command, "redis_host", "127.0.0.1") : args_.redis_host_override;
    twist_redis_port_ = args_.redis_port > 0 ? args_.redis_port : scalar_or<int>(motion_command, "redis_port", 6379);
    twist_redis_key_ = args_.redis_key_override.empty() ? scalar_or<std::string>(motion_command, "redis_key", "action_mimic_g1") : args_.redis_key_override;
    twist_input_timeout_sec_ = args_.input_timeout_sec >= 0.0 ? args_.input_timeout_sec : scalar_or<double>(motion_command, "input_timeout_sec", 2.0);
    twist_stale_sec_ = args_.stale_sec >= 0.0 ? args_.stale_sec : scalar_or<double>(motion_command, "stale_sec", 0.25);
    std::string stale_behavior = args_.stale_behavior_override.empty()
                                     ? scalar_or<std::string>(motion_command, "stale_behavior", "hold_then_stop")
                                     : args_.stale_behavior_override;
    if (twist_stale_to_default) stale_behavior = "default";
    twist_stale_behavior_ = parse_stale_behavior_value(stale_behavior);
  }

void GenericPolicyPlayer::load_spec_humanoid_gpt(const YAML::Node& y) {
    const YAML::Node runtime = y["runtime"];
    const YAML::Node reference = y["reference"];
    humanoid_gpt_obs_dim_ = configured_obs_dim_;
    humanoid_gpt_qpos_dim_ = yamlc::required_scalar<int>(y, "reference.qpos_dim");
    humanoid_gpt_qvel_dim_ = yamlc::required_scalar<int>(y, "reference.qvel_dim");
    humanoid_gpt_keypoint_count_ = yamlc::required_scalar<int>(y, "reference.keypoint_count");
    humanoid_gpt_hmcp_header_size_ = yamlc::required_scalar<int>(y, "reference.hmcp_header_size");
    if (humanoid_gpt_qpos_dim_ != protocol_->humanoid_gpt_qpos_dim ||
        humanoid_gpt_qvel_dim_ != protocol_->humanoid_gpt_qvel_dim ||
        humanoid_gpt_keypoint_count_ != protocol_->humanoid_gpt_keypoint_count ||
        humanoid_gpt_hmcp_header_size_ != protocol_->humanoid_gpt_hmcp_header_size) {
      throw std::runtime_error("Humanoid-GPT protocol dimensions in spec do not match the current UDP/NPZ parser constraints");
    }
    humanoid_gpt_input_source_ = lower(args_.input_override.empty()
                                           ? scalar_or<std::string>(reference, "source", scalar_or<std::string>(runtime, "input", "offline-npz"))
                                           : args_.input_override);
    humanoid_gpt_realtime_transport_ = lower(args_.realtime_transport_override.empty()
                                                 ? scalar_or<std::string>(reference, "realtime_transport", "udp")
                                                 : args_.realtime_transport_override);
    if (humanoid_gpt_input_source_ == "offline" || humanoid_gpt_input_source_ == "npz") {
      humanoid_gpt_input_source_ = "offline-npz";
    } else if (humanoid_gpt_input_source_ == "udp" || humanoid_gpt_input_source_ == "realtime-udp") {
      humanoid_gpt_input_source_ = "realtime";
      humanoid_gpt_realtime_transport_ = "udp";
    }
    humanoid_gpt_loop_reference_ = scalar_or<bool>(reference, "loop", false) || args_.loop;
    humanoid_gpt_require_realtime_ = scalar_or<bool>(reference, "require_realtime", false) || args_.require_realtime;
    humanoid_gpt_udp_bind_ = args_.udp_bind_override.empty() ? scalar_or<std::string>(reference, "udp_bind", "127.0.0.1") : args_.udp_bind_override;
    humanoid_gpt_udp_port_ = args_.udp_port > 0 ? args_.udp_port : scalar_or<int>(reference, "udp_port", 51234);
    humanoid_gpt_input_timeout_sec_ = args_.input_timeout_sec >= 0.0 ? args_.input_timeout_sec : scalar_or<double>(reference, "input_timeout_sec", 3.0);
    humanoid_gpt_stale_sec_ = args_.stale_sec >= 0.0 ? args_.stale_sec : scalar_or<double>(reference, "stale_sec", 0.25);
    const std::string humanoid_stale_behavior = args_.stale_behavior_override.empty()
                                                    ? scalar_or<std::string>(reference, "stale_behavior", "hold_then_stop")
                                                    : args_.stale_behavior_override;
    humanoid_gpt_stale_behavior_ = parse_stale_behavior_value(humanoid_stale_behavior);
    humanoid_gpt_convert_xml_path_ = resolve_relative(
        spec_path_, scalar_or<std::string>(reference, "convert_xml_path", "../assets/g1_mjx_track_meshless.xml"));
  }

void GenericPolicyPlayer::load_spec_sonic(const YAML::Node& y) {
    const YAML::Node runtime = y["runtime"];
    const YAML::Node reference = y["reference"];
    const YAML::Node observation = y["observation"];
    const YAML::Node provider = y["model"]["provider"];
    const YAML::Node planner = yamlc::required_node(y, "planner");
    sonic_checkpoint_ = lower(scalar_or<std::string>(runtime, "checkpoint", "default"));
    int expected_encoder_obs_dim = 0;
    if (sonic_checkpoint_ == "default") {
      expected_encoder_obs_dim = kSonicEncoderObsDim;
      sonic_observation_builder_.set_layout(SonicObservationLayout::kDefault);
    } else if (sonic_checkpoint_ == "sonic_v1_1") {
      expected_encoder_obs_dim = kSonicV11EncoderObsDim;
      sonic_observation_builder_.set_layout(SonicObservationLayout::kV11);
    } else {
      throw std::runtime_error("unsupported SONIC runtime.checkpoint: " + sonic_checkpoint_);
    }
    sonic_encoder_obs_dim_ = yamlc::required_scalar<int>(y, "observation.encoder_obs_dim");
    sonic_decoder_obs_dim_ = configured_obs_dim_;
    sonic_token_dim_ = yamlc::required_scalar<int>(y, "observation.token_dim");
    sonic_history_len_ = yamlc::required_scalar<int>(y, "observation.history_len");
    const int sonic_smpl_joint_count = yamlc::required_scalar<int>(y, "observation.smpl_joint_count");
    const int sonic_smpl_pose_count = yamlc::required_scalar<int>(y, "observation.smpl_pose_count");
    const int sonic_smpl_reference_frames = yamlc::required_scalar<int>(y, "observation.smpl_reference_frames");
    const auto sonic_zmq_protocol_versions =
        yamlc::required_vector<int>(y, "reference.zmq_protocol_versions");
    sonic_future_frames_ = yamlc::required_scalar<int>(y, "reference.future_frames");
    sonic_future_step_ = yamlc::required_scalar<int>(y, "reference.future_step");
    sonic_zmq_header_size_ = yamlc::required_scalar<int>(y, "reference.zmq_header_size");
    if (sonic_encoder_obs_dim_ != expected_encoder_obs_dim || sonic_decoder_obs_dim_ != kSonicDecoderObsDim ||
        sonic_token_dim_ != kSonicTokenDim || sonic_history_len_ != kSonicHistoryLen ||
        sonic_smpl_joint_count != kSonicSmplJointCount || sonic_smpl_pose_count != kSonicSmplPoseCount ||
        sonic_smpl_reference_frames != kSonicSmplReferenceFrames ||
        sonic_future_frames_ != kSonicReferenceFrames || sonic_future_step_ != kSonicReferenceStep ||
        sonic_zmq_header_size_ != kSonicZmqHeaderSize ||
        sonic_zmq_protocol_versions != std::vector<int>({1, 2, 3})) {
      throw std::runtime_error("SONIC spec dimensions do not match runtime.checkpoint=" + sonic_checkpoint_);
    }
    sonic_input_source_ = lower(args_.input_override.empty()
                                    ? scalar_or<std::string>(reference, "source", "offline-csv")
                                    : args_.input_override);
    sonic_pico_control_ = sonic_input_source_ == "pico" || sonic_input_source_ == "pico-zmq";
    if (sonic_input_source_ == "offline" || sonic_input_source_ == "csv") sonic_input_source_ = "offline-csv";
    if (sonic_input_source_ == "realtime" || sonic_input_source_ == "zmq" ||
        sonic_input_source_ == "realtime-zmq-v1" || sonic_input_source_ == "realtime-zmq-v3" ||
        sonic_input_source_ == "pico" || sonic_input_source_ == "pico-zmq") {
      sonic_input_source_ = "realtime-zmq";
    }
    sonic_loop_reference_ = scalar_or<bool>(reference, "loop", false) || args_.loop;
    sonic_require_realtime_ = scalar_or<bool>(reference, "require_realtime", false) || args_.require_realtime;
    sonic_zmq_endpoint_ = args_.zmq_endpoint_override.empty()
                              ? scalar_or<std::string>(reference, "zmq_endpoint", "tcp://127.0.0.1:5556")
                              : args_.zmq_endpoint_override;
    sonic_zmq_topic_ = args_.zmq_topic_override.empty()
                           ? scalar_or<std::string>(reference, "zmq_topic", "pose")
                           : args_.zmq_topic_override;
    sonic_zmq_bind_ = args_.zmq_bind_set ? args_.zmq_bind : scalar_or<bool>(reference, "zmq_bind", false);
    sonic_input_timeout_sec_ = args_.input_timeout_sec >= 0.0 ? args_.input_timeout_sec
                                                              : scalar_or<double>(reference, "input_timeout_sec", 3.0);
    sonic_stale_sec_ = args_.stale_sec >= 0.0 ? args_.stale_sec : scalar_or<double>(reference, "stale_sec", 0.1);
    sonic_stale_stop_sec_ = scalar_or<double>(reference, "stale_stop_sec", 0.5);
    sonic_startup_ramp_sec_ = scalar_or<double>(runtime, "startup_ramp_sec", 3.0);
    sonic_execution_provider_ = args_.execution_provider_override.empty()
                                    ? lower(scalar_or<std::string>(provider, "execution_provider", "cpu"))
                                    : args_.execution_provider_override;
    sonic_gpu_device_id_ = args_.gpu_device_id >= 0 ? args_.gpu_device_id : scalar_or<int>(provider, "device_id", 0);
    std::string cache = args_.trt_cache_dir_override.empty()
                            ? scalar_or<std::string>(provider, "tensorrt_cache_dir", "/tmp/action_go_sonic_trt_cache")
                            : args_.trt_cache_dir_override;
    sonic_trt_cache_dir_ = fs::path(cache);
    sonic_trt_workspace_size_ = args_.trt_workspace_size > 0 ? args_.trt_workspace_size
                                                              : scalar_or<size_t>(provider, "tensorrt_workspace_size", 1ULL << 30);
    sonic_trt_fp16_ = args_.trt_fp16_set ? args_.trt_fp16 : scalar_or<bool>(provider, "tensorrt_fp16", false);
    sonic_warmup_iters_ = args_.warmup_iters >= 0 ? args_.warmup_iters : scalar_or<int>(provider, "warmup_iters", 1);
    if (yamlc::required_scalar<bool>(planner, "enabled")) {
      sonic_planner_config_.version = yamlc::required_scalar<int>(planner, "version");
      if (yamlc::required_scalar<int>(planner, "inference_hz") != 10 ||
          yamlc::required_scalar<int>(planner, "output_hz") != 30 ||
          yamlc::required_scalar<int>(planner, "reference_hz") != 50 ||
          yamlc::required_scalar<int>(planner, "context_frames") != kSonicPlannerContextFrames ||
          yamlc::required_scalar<int>(planner, "context_qpos_dim") != kSonicPlannerQposDim ||
          yamlc::required_scalar<int>(planner, "mode_count") != kSonicPlannerModeCount) {
        throw std::runtime_error("SONIC planner timing or tensor dimensions differ from the V2 contract");
      }
      sonic_planner_config_.lookahead_steps = yamlc::required_scalar<int>(planner, "lookahead_steps");
      sonic_planner_config_.blend_frames = yamlc::required_scalar<int>(planner, "blend_frames");
      sonic_planner_config_.min_output_frames = yamlc::required_scalar<int>(planner, "min_output_frames");
      sonic_planner_config_.max_output_frames = yamlc::required_scalar<int>(planner, "max_output_frames");
      sonic_planner_config_.default_height = yamlc::required_scalar<float>(planner, "default_height");
      sonic_planner_config_.random_seed = yamlc::required_scalar<int64_t>(planner, "random_seed");
      const auto allowed = yamlc::required_vector_size<int64_t>(planner, "allowed_pred_num_tokens", 11);
      std::copy(allowed.begin(), allowed.end(), sonic_planner_config_.allowed_pred_num_tokens.begin());
      sonic_planner_ = SonicPlanner(sonic_planner_config_);
    } else if (sonic_pico_control_) {
      throw std::runtime_error("SONIC pico-zmq input requires planner.enabled=true");
    }
    (void)observation;
  }

void GenericPolicyPlayer::post_load_protomotions() {
    onnx_.require_inputs(proto_input_names_);
    onnx_.require_outputs({"joint_pos_targets", "stiffness_targets", "damping_targets"});
    for (const auto& name : onnx_.input_names()) {
      if (proto_name_to_key_.find(name) == proto_name_to_key_.end()) {
        throw std::runtime_error("ProtoMotions params missing semantic key for ONNX input: " + name);
      }
    }
  }

void GenericPolicyPlayer::post_load_twist() {
    twist_obs_dim_ = onnx_.inferred_obs_dim(twist_obs_dim_);
    if (twist_obs_dim_ != twist_input_dim_) {
      throw std::runtime_error("TWIST ONNX obs dim=" + std::to_string(twist_obs_dim_) +
                               " (expected " + std::to_string(twist_input_dim_) + " from spec)");
    }
    obs_input_name_ = obs_input_name_.empty() ? "obs" : obs_input_name_;
    action_output_name_ = action_output_name_.empty() ? "action" : action_output_name_;
    onnx_.require_inputs({obs_input_name_});
    onnx_.require_outputs({action_output_name_});
    setup_twist_input();
  }

void GenericPolicyPlayer::post_load_humanoid_gpt() {
    humanoid_gpt_obs_dim_ = onnx_.inferred_obs_dim(humanoid_gpt_obs_dim_);
    if (configured_obs_dim_ > 0 && humanoid_gpt_obs_dim_ != configured_obs_dim_) {
      throw std::runtime_error("Humanoid-GPT ONNX obs dim=" + std::to_string(humanoid_gpt_obs_dim_) +
                               " (expected " + std::to_string(configured_obs_dim_) + " from spec)");
    }
    obs_input_name_ = obs_input_name_.empty() ? "obs" : obs_input_name_;
    action_output_name_ = action_output_name_.empty() ? "continuous_actions" : action_output_name_;
    onnx_.require_inputs({obs_input_name_});
    onnx_.require_outputs({action_output_name_});
    setup_humanoid_gpt_reference();
  }

void GenericPolicyPlayer::post_load_sonic() {
    if (!fs::exists(sonic_encoder_path_)) throw std::runtime_error("SONIC encoder not found: " + sonic_encoder_path_.string());
    if (!fs::exists(sonic_decoder_path_)) throw std::runtime_error("SONIC decoder not found: " + sonic_decoder_path_.string());
    OnnxRunner::LoadOptions options;
    options.threads = onnx_threads_;
    options.execution_provider = sonic_execution_provider_;
    options.device_id = sonic_gpu_device_id_;
    options.tensorrt_cache_dir = sonic_trt_cache_dir_;
    options.tensorrt_workspace_size = sonic_trt_workspace_size_;
    options.tensorrt_fp16 = sonic_trt_fp16_;
    sonic_encoder_.load(sonic_encoder_path_, options);
    sonic_decoder_.load(sonic_decoder_path_, options);
    sonic_encoder_.require_inputs({"obs_dict"});
    sonic_encoder_.require_outputs({"encoded_tokens"});
    sonic_decoder_.require_inputs({"obs_dict"});
    sonic_decoder_.require_outputs({"action"});
    if (sonic_encoder_.inferred_obs_dim(0) != sonic_encoder_obs_dim_ ||
        sonic_decoder_.inferred_obs_dim(0) != sonic_decoder_obs_dim_) {
      throw std::runtime_error("SONIC ONNX input dimensions do not match the spec");
    }
    if (sonic_pico_control_) {
      if (!fs::exists(sonic_planner_path_)) {
        throw std::runtime_error("SONIC planner not found: " + sonic_planner_path_.string());
      }
      sonic_planner_runner_.load(sonic_planner_path_, options);
      sonic_planner_runner_.require_inputs({
          "context_mujoco_qpos", "target_vel", "mode", "movement_direction",
          "facing_direction", "random_seed", "has_specific_target",
          "specific_target_positions", "specific_target_headings",
          "allowed_pred_num_tokens", "height"});
      sonic_planner_runner_.require_outputs({"mujoco_qpos", "num_pred_frames"});
    }
    setup_sonic_reference();
  }

void GenericPolicyPlayer::load_protomotions_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("ProtoMotions params not found: " + params_path_.string());
    if (trajectory_path_.empty()) throw std::runtime_error("ProtoMotions requires reference.path or --trajectory");
    if (!fs::exists(trajectory_path_)) throw std::runtime_error("ProtoMotions motion not found: " + trajectory_path_.string());

    const YAML::Node y = YAML::LoadFile(params_path_.string());
    const YAML::Node runtime = y["_runtime"];
    proto_num_dofs_ = yamlc::required_scalar<int>(y, "robot.motor_count");
    proto_num_bodies_ = yamlc::required_scalar<int>(y, "reference.num_bodies");
    proto_anchor_idx_ = yamlc::required_scalar<int>(y, "reference.anchor_body_index");
    proto_root_idx_ = yamlc::required_scalar<int>(y, "reference.root_body_index");
    if (proto_num_dofs_ != motor_count_) {
      throw std::runtime_error("ProtoMotions robot.motor_count in params must match spec robot.motor_count");
    }
    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.joint_order", motor_count_);
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);

    proto_future_steps_.clear();
    for (const auto& v : yamlc::required_node(y, "motion.future_step_indices")) proto_future_steps_.push_back(v.as<int>());

    proto_input_names_ = runtime["onnx_in_names"] ? read_vec<std::string>(runtime, "onnx_in_names") : expected_input_names_;
    if (proto_input_names_.empty()) throw std::runtime_error("ProtoMotions requires model.input_names or _runtime.onnx_in_names");
    proto_name_to_key_.clear();
    for (const auto& kv : runtime["onnx_name_to_in_key"]) {
      proto_name_to_key_[kv.first.as<std::string>()] = kv.second.as<std::string>();
    }

    proto_motion_.load(trajectory_path_, motor_count_);
    if (proto_motion_.num_bodies != proto_num_bodies_) {
      throw std::runtime_error("ProtoMotions motion body count != YAML num_bodies");
    }
    proto_prev_actions_.assign(motor_count_, 0.f);
    proto_heading_uninitialized_ = true;
    proto_frame_ = 0;
  }

void GenericPolicyPlayer::load_twist_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("TWIST params not found: " + params_path_.string());
    const YAML::Node y = YAML::LoadFile(params_path_.string());
    const int num_actions = yamlc::required_scalar<int>(y, "policy.num_actions");
    const int params_obs_dim = yamlc::required_scalar<int>(y, "policy.num_obs");
    const int params_history_len = yamlc::required_scalar<int>(y, "observation.history_len");
    if (num_actions != twist_policy_actions_) {
      throw std::runtime_error("TWIST policy.num_actions does not match spec action.action_dim");
    }
    if (params_obs_dim != twist_input_dim_) {
      throw std::runtime_error("TWIST policy.num_obs does not match spec observation.input_dim");
    }
    if (params_history_len != twist_history_len_) {
      throw std::runtime_error("TWIST observation.history_len in params does not match spec");
    }
    twist_obs_dim_ = params_obs_dim;

    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.joint_order", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    twist_body_motor_idx_ = yamlc::required_vector_size<int>(y, "robot.body_motor_idx", twist_policy_actions_);
    twist_wrist_motor_idx_ = yamlc::required_vector_size<int>(y, "robot.wrist_motor_idx", twist_wrist_dim_);
    twist_ankle_policy_idx_ = yamlc::required_vector_size<int>(y, "observation.ankle_policy_idx", 4);
    twist_default_dof_pos_ = yamlc::required_vector_size<float>(y, "robot.default_angles", twist_policy_actions_);
    twist_body_kp_ = yamlc::required_vector_size<float>(y, "robot.body_kp", twist_policy_actions_);
    twist_body_kd_ = yamlc::required_vector_size<float>(y, "robot.body_kd", twist_policy_actions_);
    twist_wrist_default_ = yamlc::required_vector_size<float>(y, "robot.wrist_default_angles", twist_wrist_dim_);
    twist_wrist_kp_ = yamlc::required_vector_size<float>(y, "robot.wrist_kp", twist_wrist_dim_);
    twist_wrist_kd_ = yamlc::required_vector_size<float>(y, "robot.wrist_kd", twist_wrist_dim_);
    twist_default_mimic_obs_ =
        to_float_array<kTwistProtocolMimicDim>(yamlc::required_vector<float>(y, "motion.default_mimic_obs"),
                                           "motion.default_mimic_obs");
    twist_ang_vel_scale_ = yamlc::required_scalar<float>(y, "observation.ang_vel_scale");
    twist_dof_pos_scale_ = yamlc::required_scalar<float>(y, "observation.dof_pos_scale");
    twist_dof_vel_scale_ = yamlc::required_scalar<float>(y, "observation.dof_vel_scale");
    if (!std::isfinite(twist_action_scale_)) twist_action_scale_ = yamlc::required_scalar<float>(y, "action.scale");

    check_size("TWIST body_motor_idx", twist_policy_actions_, twist_body_motor_idx_.size());
    check_size("TWIST default_dof_pos", twist_policy_actions_, twist_default_dof_pos_.size());
    check_size("TWIST body_kp", twist_policy_actions_, twist_body_kp_.size());
    check_size("TWIST body_kd", twist_policy_actions_, twist_body_kd_.size());
    check_size("TWIST wrist_motor_idx", static_cast<size_t>(twist_wrist_dim_), twist_wrist_motor_idx_.size());
    check_size("TWIST wrist_default", static_cast<size_t>(twist_wrist_dim_), twist_wrist_default_.size());
    check_size("TWIST wrist_kp", static_cast<size_t>(twist_wrist_dim_), twist_wrist_kp_.size());
    check_size("TWIST wrist_kd", static_cast<size_t>(twist_wrist_dim_), twist_wrist_kd_.size());
    for (int idx : twist_body_motor_idx_) {
      if (idx < 0 || idx >= motor_count_) throw std::runtime_error("TWIST body_motor_idx out of range");
    }
    for (int idx : twist_wrist_motor_idx_) {
      if (idx < 0 || idx >= motor_count_) throw std::runtime_error("TWIST wrist_motor_idx out of range");
    }
    for (int idx : twist_ankle_policy_idx_) {
      if (idx < 0 || idx >= twist_policy_actions_) throw std::runtime_error("TWIST ankle_policy_idx out of range");
    }
    twist_offline_.set_default_frame(twist_default_mimic_obs_);
    reset_twist_controller();
  }

void GenericPolicyPlayer::load_humanoid_gpt_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("Humanoid-GPT params not found: " + params_path_.string());
    const YAML::Node y = YAML::LoadFile(params_path_.string());
    const int num_actions = yamlc::required_scalar<int>(y, "policy.num_actions");
    humanoid_gpt_obs_dim_ = yamlc::required_scalar<int>(y, "policy.num_obs");
    const int num_qpos = yamlc::required_scalar<int>(y, "reference.num_qpos");
    const int num_qvel = yamlc::required_scalar<int>(y, "reference.num_qvel");
    if (num_actions != motor_count_) {
      throw std::runtime_error("Humanoid-GPT policy.num_actions must match spec robot.motor_count");
    }
    if (configured_obs_dim_ > 0 && humanoid_gpt_obs_dim_ != configured_obs_dim_) {
      throw std::runtime_error("Humanoid-GPT policy.num_obs does not match spec observation.obs_dim");
    }
    if (num_qpos != humanoid_gpt_qpos_dim_) {
      throw std::runtime_error("Humanoid-GPT reference.num_qpos in params does not match spec reference.qpos_dim");
    }
    if (num_qvel != humanoid_gpt_qvel_dim_) {
      throw std::runtime_error("Humanoid-GPT reference.num_qvel in params does not match spec reference.qvel_dim");
    }

    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.joint_order", motor_count_);
    humanoid_gpt_keypoint_names_ = yamlc::required_vector_size<std::string>(y, "reference.keypoint_order", humanoid_gpt_keypoint_count_);
    default_q_ = yamlc::required_vector_size<float>(y, "robot.default_angles", motor_count_);
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
    last_action_.assign(motor_count_, 0.f);
    humanoid_gpt_ref_curr_ = default_humanoid_gpt_reference();
    humanoid_gpt_ref_next_ = humanoid_gpt_ref_curr_;
    humanoid_gpt_have_realtime_ref_ = false;
    humanoid_gpt_stale_gate_.reset();
  }

void GenericPolicyPlayer::load_sonic_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("SONIC params not found: " + params_path_.string());
    const YAML::Node y = YAML::LoadFile(params_path_.string());
    if (yamlc::required_scalar<int>(y, "policy.num_actions") != motor_count_ ||
        yamlc::required_scalar<int>(y, "policy.encoder_obs_dim") != sonic_encoder_obs_dim_ ||
        yamlc::required_scalar<int>(y, "policy.decoder_obs_dim") != kSonicDecoderObsDim ||
        yamlc::required_scalar<int>(y, "policy.token_dim") != kSonicTokenDim) {
      throw std::runtime_error("SONIC params dimensions do not match the released checkpoint");
    }
    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.joint_order", motor_count_);
    default_q_ = yamlc::required_vector_size<float>(y, "robot.default_angles", motor_count_);
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
    action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
    const auto policy_to_hardware = yamlc::required_vector_size<int>(y, "robot.isaaclab_to_hardware", motor_count_);
    const auto hardware_to_policy = yamlc::required_vector_size<int>(y, "robot.hardware_to_isaaclab", motor_count_);
    if (!std::equal(policy_to_hardware.begin(), policy_to_hardware.end(), sonic_policy_to_hardware().begin()) ||
        !std::equal(hardware_to_policy.begin(), hardware_to_policy.end(), sonic_hardware_to_policy().begin())) {
      throw std::runtime_error("SONIC joint remap differs from the official deployment contract");
    }
    last_action_.assign(motor_count_, 0.f);
  }

fs::path GenericPolicyPlayer::ensure_twist_motion_cache() {
    if (twist_motion_pkl_path_.empty()) throw std::runtime_error("TWIST offline-pkl requires motion_command.pkl or --motion-pkl");
    const fs::path pkl = fs::absolute(twist_motion_pkl_path_).lexically_normal();
    if (!fs::exists(pkl)) throw std::runtime_error("TWIST motion pkl not found: " + pkl.string());

    fs::path cache = twist_motion_cache_path_;
    if (cache.empty()) {
      const auto h = std::hash<std::string>{}(pkl.string() + twist_steps_ + std::to_string(max_frames_));
      cache = fs::temp_directory_path() / ("action_go_twist_" + std::to_string(h) + ".mimic.csv");
    }
    cache = fs::absolute(cache).lexically_normal();
    if (fs::exists(cache) && !twist_rebuild_motion_cache_) return cache;

    fs::path script = twist_prepare_script_.empty()
                          ? (action_go_root_from_exe() / "tools" / "runtime" / "twist_prepare_motion.py")
                          : resolve_relative(spec_path_, twist_prepare_script_);
    if (!fs::exists(script)) throw std::runtime_error("twist_prepare_motion.py not found: " + script.string());
    if (cache.has_parent_path()) fs::create_directories(cache.parent_path());
    std::string cmd = "python3 " + shell_quote(script.string()) +
                      " --twist-root " + shell_quote(twist_root_) +
                      " --motion-pkl " + shell_quote(pkl.string()) +
                      " --out " + shell_quote(cache.string()) +
                      " --steps " + shell_quote(twist_steps_);
    if (max_frames_ > 0) cmd += " --max-frames " + std::to_string(max_frames_);
    std::cout << player_log_prefix() << " preparing TWIST motion cache: " << cmd << "\n";
    const int ret = std::system(cmd.c_str());
    if (ret != 0 || !fs::exists(cache)) {
      throw std::runtime_error("TWIST motion cache preparation failed ret=" + std::to_string(ret));
    }
    return cache;
  }

void GenericPolicyPlayer::setup_twist_input() {
    if (twist_input_source_ == "realtime") {
      if (twist_realtime_transport_ == "udp") {
        twist_udp_.open_socket(twist_udp_bind_, twist_udp_port_);
      } else if (twist_realtime_transport_ == "zmq") {
        twist_zmq_.open_socket(twist_zmq_endpoint_, twist_zmq_topic_, twist_zmq_bind_);
      } else if (twist_realtime_transport_ == "redis") {
        twist_redis_.open_socket(twist_redis_host_, twist_redis_port_, twist_redis_key_);
      } else {
        throw std::runtime_error("TWIST realtime_transport must be udp, zmq, or redis");
      }
      return;
    }

    fs::path cache;
    if (twist_input_source_ == "offline-pkl") {
      cache = ensure_twist_motion_cache();
    } else if (twist_input_source_ == "offline-cache") {
      if (twist_motion_cache_path_.empty()) throw std::runtime_error("TWIST offline-cache requires motion_command.cache or --motion-cache");
      cache = twist_motion_cache_path_;
    } else {
      throw std::runtime_error("TWIST input source must be offline-pkl, offline-cache, realtime, udp, zmq, or redis");
    }
    twist_offline_.load_csv(cache);
    std::cout << player_log_prefix() << " loaded TWIST motion cache frames=" << twist_offline_.size()
              << " dt=" << twist_offline_.dt()
              << " duration=" << twist_offline_.duration_sec() << "s\n";
  }

void GenericPolicyPlayer::setup_humanoid_gpt_reference() {
    if (humanoid_gpt_input_source_ == "offline-npz") {
      if (trajectory_path_.empty()) throw std::runtime_error("Humanoid-GPT offline-npz requires reference.path or --trajectory");
      if (!fs::exists(trajectory_path_)) throw std::runtime_error("Humanoid-GPT reference cache not found: " + trajectory_path_.string());
      humanoid_gpt_reference_cache_.load(trajectory_path_);
      humanoid_gpt_ref_frame_ = 0;
      humanoid_gpt_ref_curr_ = humanoid_gpt_reference_cache_.frame_at(0);
      humanoid_gpt_ref_next_ = humanoid_gpt_reference_cache_.frame_at(std::min(1, humanoid_gpt_reference_cache_.num_frames() - 1));
      std::cout << player_log_prefix() << " loaded Humanoid-GPT reference cache frames="
                << humanoid_gpt_reference_cache_.num_frames()
                << " hz=" << humanoid_gpt_reference_cache_.frequency_hz()
                << " duration=" << humanoid_gpt_reference_cache_.duration_sec()
                << "s path=" << trajectory_path_.string() << "\n";
      return;
    }
    if (humanoid_gpt_input_source_ == "realtime") {
      if (humanoid_gpt_realtime_transport_ != "udp") {
        throw std::runtime_error("Humanoid-GPT realtime_transport currently supports udp only");
      }
      humanoid_gpt_udp_.open_socket(humanoid_gpt_udp_bind_, humanoid_gpt_udp_port_);
      if (!fs::exists(humanoid_gpt_convert_xml_path_)) {
        throw std::runtime_error("Humanoid-GPT realtime FK XML not found: " + humanoid_gpt_convert_xml_path_.string());
      }
      humanoid_gpt_converter_.load(humanoid_gpt_convert_xml_path_,
                                   joint_order_,
                                   humanoid_gpt_keypoint_names_,
                                   static_cast<float>(1.0 / static_cast<double>(std::max(1, inference_hz_))));
      return;
    }
    throw std::runtime_error("Humanoid-GPT reference.source must be offline-npz or realtime");
  }

void GenericPolicyPlayer::setup_sonic_reference() {
    if (sonic_input_source_ == "offline-csv") {
      if (trajectory_path_.empty()) throw std::runtime_error("SONIC offline-csv requires reference.path or --trajectory");
      sonic_reference_cache_.load_csv_directory(trajectory_path_);
      std::cout << player_log_prefix() << " loaded SONIC reference frames=" << sonic_reference_cache_.size()
                << " path=" << trajectory_path_.string() << "\n";
      return;
    }
    if (sonic_input_source_ == "realtime-zmq") {
      sonic_zmq_.open_socket(sonic_zmq_endpoint_, sonic_zmq_topic_, sonic_zmq_bind_);
      return;
    }
    throw std::runtime_error("SONIC reference.source must be offline-csv or realtime-zmq");
  }

void GenericPolicyPlayer::schema_summary_protomotions() {
    schema_summary_common();
    std::cout << "  protomotions_contract inputs=" << proto_input_names_.size()
              << " future_steps=" << proto_future_steps_.size() << "\n";
  }

void GenericPolicyPlayer::schema_summary_twist() {
    schema_summary_common();
    std::cout << "  twist_contract mimic_dim=" << twist_full_mimic_dim_
              << " policy_mimic_dim=" << twist_policy_mimic_dim_
              << " proprio_dim=" << twist_proprio_dim_
              << " wrist_dim=" << twist_wrist_dim_ << "\n";
  }

void GenericPolicyPlayer::schema_summary_humanoid_gpt() {
    schema_summary_common();
    std::cout << "  humanoid_gpt_contract qpos_dim=" << humanoid_gpt_qpos_dim_
              << " qvel_dim=" << humanoid_gpt_qvel_dim_
              << " keypoint_count=" << humanoid_gpt_keypoint_count_
              << " hmcp_header_size=" << humanoid_gpt_hmcp_header_size_ << "\n";
  }

void GenericPolicyPlayer::schema_summary_sonic() {
    schema_summary_common();
    std::cout << "  sonic_contract checkpoint=" << sonic_checkpoint_
              << " encoder_obs_dim=" << sonic_encoder_obs_dim_
              << " token_dim=" << sonic_token_dim_
              << " decoder_obs_dim=" << sonic_decoder_obs_dim_
              << " history_len=" << sonic_history_len_
              << " reference=" << sonic_future_frames_ << "xstep" << sonic_future_step_
              << " provider=" << sonic_decoder_.selected_execution_provider() << "\n";
  }

void GenericPolicyPlayer::check_only_protomotions() {
    RobotState s;
    proto_motion_.dof_at(0, s.q, s.dq);
    s.quat = {1.f, 0.f, 0.f, 0.f};
    s.gyro = {0.f, 0.f, 0.f};
    auto target = protomotions_infer(s);
    auto mm = std::minmax_element(target.begin(), target.end());
    std::cout << "  protomotions motion=" << trajectory_path_.string()
              << " frames=" << proto_motion_.num_frames
              << " dt=" << proto_motion_.control_dt
              << " bodies=" << proto_motion_.num_bodies
              << " first_target_range=[" << *mm.first << "," << *mm.second << "]\n";
  }

void GenericPolicyPlayer::check_only_twist() {
    if (twist_input_source_ == "realtime" && twist_require_realtime_ && !wait_first_twist_realtime()) {
      throw std::runtime_error("timeout waiting for first TWIST realtime mimic_obs packet during check-only");
    }
    reset_twist_controller();
    RobotState s;
    s.q.assign(motor_count_, 0.f);
    s.dq.assign(motor_count_, 0.f);
    s.quat = {1.f, 0.f, 0.f, 0.f};
    s.gyro = {0.f, 0.f, 0.f};
    std::vector<float> q, kp, kd, raw_action;
    const auto status = build_twist_target(s, 0, q, kp, kd, raw_action);
    if (status != TwistTargetStatus::kCommand) {
      throw std::runtime_error("TWIST check-only motion source is already finished");
    }
    const auto am = std::minmax_element(raw_action.begin(), raw_action.end());
    const auto qm = std::minmax_element(q.begin(), q.end());
    std::cout << "  twist input=" << twist_input_source_
              << " transport=" << twist_realtime_transport_
              << " obs_dim=" << twist_obs_dim_
              << " history_len=" << twist_history_len_
              << " action_dim=" << raw_action.size()
              << " action_range=[" << *am.first << "," << *am.second << "]"
              << " target_q_range=[" << *qm.first << "," << *qm.second << "]\n";
  }

void GenericPolicyPlayer::check_only_humanoid_gpt() {
    RobotState s;
    s.q = default_q_;
    s.dq.assign(motor_count_, 0.f);
    s.quat = {1.f, 0.f, 0.f, 0.f};
    s.gyro = {0.f, 0.f, 0.f};
    std::vector<float> q, kp, kd, raw_action;
    if (humanoid_gpt_input_source_ == "realtime") {
      humanoid_gpt_ref_curr_ = default_humanoid_gpt_reference();
      humanoid_gpt_ref_next_ = humanoid_gpt_ref_curr_;
      const auto obs = build_humanoid_gpt_obs(s, humanoid_gpt_ref_curr_, humanoid_gpt_ref_next_);
      raw_action = humanoid_gpt_infer(obs);
      q.assign(motor_count_, 0.f);
      kp = kp_;
      kd = kd_;
      for (int i = 0; i < motor_count_; ++i) {
        q[i] = std::clamp(default_q_[i] + raw_action[i] * action_scale_[i],
                          lower_limits_[i], upper_limits_[i]);
      }
    } else {
      const auto status = build_humanoid_gpt_target(s, 0, q, kp, kd, raw_action);
      if (status != HumanoidGptTargetStatus::kCommand) {
        throw std::runtime_error("Humanoid-GPT check-only reference source is unavailable");
      }
    }
    const auto am = std::minmax_element(raw_action.begin(), raw_action.end());
    const auto qm = std::minmax_element(q.begin(), q.end());
    std::cout << "  humanoid_gpt input=" << humanoid_gpt_input_source_
              << " transport=" << humanoid_gpt_realtime_transport_
              << " obs_dim=" << humanoid_gpt_obs_dim_
              << " action_dim=" << raw_action.size()
              << " action_range=[" << *am.first << "," << *am.second << "]"
              << " target_q_range=[" << *qm.first << "," << *qm.second << "]";
    if (humanoid_gpt_input_source_ == "offline-npz") {
      std::cout << " frames=" << humanoid_gpt_reference_cache_.num_frames()
                << " hz=" << humanoid_gpt_reference_cache_.frequency_hz();
    }
    std::cout << "\n";
  }

void GenericPolicyPlayer::check_only_sonic() {
    if (sonic_input_source_ == "realtime-zmq" && sonic_require_realtime_ && !wait_first_sonic_realtime()) {
      throw std::runtime_error("timeout waiting for first SONIC realtime ZMQ pose packet during check-only");
    }
    RobotState state;
    state.q = default_q_;
    state.dq.assign(motor_count_, 0.f);
    state.quat = {1.f, 0.f, 0.f, 0.f};
    state.gyro = {0.f, 0.f, 0.f};
    if (sonic_pico_control_) {
      sonic_planner_.initialize(sonic_planner_runner_, state.q);
      sonic_use_planner_ = true;
      sonic_reference_cache_.replace(sonic_planner_.reference_frames());
    } else if (sonic_input_source_ == "realtime-zmq") {
      const auto sample = sonic_zmq_.current_sample(sonic_stale_sec_);
      if (sample.frames.empty()) {
        std::vector<SonicReferenceFrame> frames(
            static_cast<size_t>((sonic_future_frames_ - 1) * sonic_future_step_ + 1));
        for (size_t i = 0; i < frames.size(); ++i) frames[i].frame_index = static_cast<int64_t>(i);
        sonic_reference_cache_.replace(std::move(frames));
      } else {
        sonic_reference_cache_.replace(sample.frames);
      }
    }
    reset_sonic_runtime(state);
    std::vector<float> target, action;
    const std::string saved_input_source = sonic_input_source_;
    if (saved_input_source == "realtime-zmq") {
      sonic_input_source_ = "offline-csv";
      sonic_reference_frame_ = 0;
    }
    if (!build_sonic_target(state, 0, target, action)) throw std::runtime_error("SONIC check-only could not build a target");
    sonic_input_source_ = saved_input_source;
    if (action.empty() || target.empty()) throw std::runtime_error("SONIC check-only produced no action or target");
    const auto action_range = std::minmax_element(action.begin(), action.end());
    const auto target_range = std::minmax_element(target.begin(), target.end());
    std::cout << "  sonic input=" << sonic_input_source_
              << " checkpoint=" << sonic_checkpoint_
              << " provider=" << sonic_decoder_.selected_execution_provider()
              << " planner=" << (sonic_pico_control_ ? "loaded" : "disabled")
              << " encoder_obs_dim=" << sonic_encoder_obs_dim_
              << " decoder_obs_dim=" << sonic_decoder_obs_dim_
              << " action_range=[" << *action_range.first << "," << *action_range.second << "]"
              << " target_q_range=[" << *target_range.first << "," << *target_range.second << "]\n";
  }

bool GenericPolicyPlayer::wait_first_sonic_realtime() {
    if (!sonic_pico_control_) return sonic_zmq_.wait_first(sonic_input_timeout_sec_);
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(sonic_input_timeout_sec_));
    while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
      const auto control = sonic_zmq_.current_control_sample();
      if (control.present && control.command.start && !control.command.stop) {
        if (control.command.planner) {
          if (sonic_zmq_.current_planner_sample(sonic_stale_sec_).freshness == SampleFreshness::kFresh) return true;
        } else {
          const auto pose = sonic_zmq_.current_sample(sonic_stale_sec_);
          if (!pose.frames.empty() && pose.freshness == SampleFreshness::kFresh) return true;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
  }

SonicRobotFrame GenericPolicyPlayer::sonic_robot_frame(const GenericPolicyPlayer::RobotState& state) const {
    if (state.q.size() < motor_count_ || state.dq.size() < motor_count_) {
      throw std::runtime_error("SONIC LowState motor count is too small");
    }
    for (int i = 0; i < motor_count_; ++i) {
      if (!std::isfinite(state.q[i]) || !std::isfinite(state.dq[i])) {
        throw std::runtime_error("SONIC LowState contains NaN/Inf");
      }
      if (std::abs(state.dq[i]) > 35.f) {
        throw std::runtime_error("SONIC LowState joint velocity exceeds 35 rad/s at motor " + std::to_string(i));
      }
    }
    SonicRobotFrame frame;
    frame.base_quat_wxyz = state.quat;
    frame.base_ang_vel = state.gyro;
    frame.joint_pos = sonic_robot_to_policy(state.q, default_q_, true);
    frame.joint_vel = sonic_robot_to_policy(state.dq, default_q_, false);
    std::copy_n(last_action_.begin(), motor_count_, frame.last_action.begin());
    return frame;
  }

void GenericPolicyPlayer::reset_sonic_runtime(const GenericPolicyPlayer::RobotState& state) {
    last_action_.assign(motor_count_, 0.f);
    if (sonic_pico_control_ && sonic_use_planner_ && sonic_planner_.initialized()) {
      sonic_planner_.rewind();
      sonic_reference_cache_.replace(sonic_planner_.reference_frames());
    }
    sonic_reference_frame_ = sonic_input_source_ == "realtime-zmq"
                                 ? sonic_reference_cache_.first_frame_index() : 0;
    sonic_realtime_base_frame_ = sonic_input_source_ == "realtime-zmq"
                                    ? sonic_reference_cache_.first_frame_index() : 0;
    sonic_realtime_frame_step_ = sonic_input_source_ == "realtime-zmq"
                                    ? sonic_reference_cache_.frame_step() : 1;
    sonic_realtime_heading_reset_ = false;
    sonic_stale_gate_.reset();
    sonic_stale_events_ = 0;
    sonic_stale_hold_ticks_ = 0;
    sonic_stale_stop_events_ = 0;
    sonic_hold_last_target_ = false;
    const auto robot = sonic_robot_frame(state);
    sonic_observation_builder_.reset(robot, sonic_reference_cache_.frame_clamped(0));
  }

void GenericPolicyPlayer::require_sonic_realtime_fresh(const char* phase, bool revoke_ready) {
    if (sonic_input_source_ != "realtime-zmq") return;
    double age_sec = std::numeric_limits<double>::infinity();
    bool fresh = false;
    std::string source = "sonic_zmq_pose";
    if (sonic_pico_control_) {
      const auto control = sonic_zmq_.current_control_sample();
      if (control.present && control.command.start && !control.command.stop) {
        if (control.command.planner) {
          const auto planner = sonic_zmq_.current_planner_sample(sonic_stale_sec_);
          age_sec = planner.age_sec;
          fresh = planner.freshness == SampleFreshness::kFresh;
          source = "sonic_zmq_planner";
        } else {
          const auto pose = sonic_zmq_.current_sample(sonic_stale_sec_);
          age_sec = pose.age_sec;
          fresh = !pose.frames.empty() && pose.freshness == SampleFreshness::kFresh;
        }
      }
    } else {
      const auto pose = sonic_zmq_.current_sample(sonic_stale_sec_);
      age_sec = pose.age_sec;
      fresh = !pose.frames.empty() && pose.freshness == SampleFreshness::kFresh;
    }
    if (fresh) return;
    if (revoke_ready && !ready_file_.empty()) {
      std::error_code error;
      fs::remove(ready_file_, error);
    }
    event_log_.log("standby_input_lost",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_string("source", source),
                    event_string("phase", phase),
                    event_number("age_sec", age_sec)});
    throw std::runtime_error(std::string("SONIC realtime reference is not fresh during ") + phase);
  }

bool GenericPolicyPlayer::prepare_sonic_pico_source(const GenericPolicyPlayer::RobotState& state,
                                                     bool require_started) {
    const auto control = sonic_zmq_.current_control_sample();
    if (!control.present || !control.command.start || control.command.stop) {
      if (control.present && control.command.stop) sonic_remote_stop_ = true;
      return !require_started;
    }
    sonic_last_control_sequence_ = control.sequence;
    sonic_use_planner_ = control.command.planner;
    if (sonic_use_planner_) {
      const auto sample = sonic_zmq_.current_planner_sample(sonic_stale_sec_);
      if (sample.freshness != SampleFreshness::kFresh) return false;
      if (!sonic_planner_.initialized()) {
        sonic_planner_.initialize(sonic_planner_runner_, state.q);
      }
      const auto now = std::chrono::steady_clock::now();
      if (sonic_planner_.should_replan(sample.command, now)) {
        sonic_planner_.replan(sonic_planner_runner_, sample.command, now);
      }
      sonic_reference_cache_.replace(sonic_planner_.reference_frames());
      sonic_reference_frame_ = sonic_planner_.current_frame();
      sonic_realtime_base_frame_ = 0;
      sonic_realtime_frame_step_ = 1;
      sonic_last_planner_sequence_ = sample.sequence;
    } else {
      const auto sample = sonic_zmq_.current_sample(sonic_stale_sec_);
      if (sample.frames.empty() || sample.freshness != SampleFreshness::kFresh) return false;
      sonic_reference_cache_.replace(sample.frames);
      sonic_realtime_base_frame_ = sample.frames.front().frame_index;
      sonic_reference_frame_ = sonic_realtime_base_frame_;
      sonic_realtime_frame_step_ = sample.frames.size() > 1
          ? sample.frames[1].frame_index - sample.frames[0].frame_index : 1;
      sonic_last_realtime_sequence_ = sample.sequence;
    }
    sonic_realtime_heading_reset_ = true;
    return true;
  }

bool GenericPolicyPlayer::update_sonic_reference(const GenericPolicyPlayer::RobotState& state,
                                                  int64_t policy_step) {
    if (sonic_input_source_ != "realtime-zmq") return true;
    if (sonic_pico_control_) {
      const auto control = sonic_zmq_.current_control_sample();
      if (!control.present || control.command.stop || !control.command.start) {
        if (control.present && control.command.stop) sonic_remote_stop_ = true;
        return false;
      }
      const bool mode_changed = control.sequence != sonic_last_control_sequence_ &&
                                control.command.planner != sonic_use_planner_;
      sonic_last_control_sequence_ = control.sequence;
      if (mode_changed) {
        sonic_use_planner_ = control.command.planner;
        if (sonic_use_planner_) sonic_planner_ = SonicPlanner(sonic_planner_config_);
        if (!prepare_sonic_pico_source(state, true)) {
          sonic_hold_last_target_ = true;
          return true;
        }
        sonic_observation_builder_.reset_heading(sonic_robot_frame(state),
                                                  sonic_reference_cache_.frame_clamped(0));
      }

      SampleFreshness freshness = SampleFreshness::kMissing;
      if (sonic_use_planner_) {
        freshness = sonic_zmq_.current_planner_sample(sonic_stale_sec_).freshness;
      } else {
        freshness = sonic_zmq_.current_sample(sonic_stale_sec_).freshness;
      }
      const auto decision = sonic_stale_gate_.decide(freshness, StaleBehavior::kHoldThenStop,
                                                     sonic_stale_stop_sec_, std::chrono::steady_clock::now());
      if (decision.new_stale_event) ++sonic_stale_events_;
      if (decision.action == StaleDecisionAction::kHoldLast) {
        ++sonic_stale_hold_ticks_;
        sonic_hold_last_target_ = true;
        return true;
      }
      if (decision.action == StaleDecisionAction::kStop) {
        ++sonic_stale_stop_events_;
        throw std::runtime_error("SONIC PICO input remained stale beyond the hold window");
      }

      if (sonic_use_planner_) {
        const auto sample = sonic_zmq_.current_planner_sample(sonic_stale_sec_);
        const auto now = std::chrono::steady_clock::now();
        if (!sonic_planner_.initialized()) {
          sonic_planner_.initialize(sonic_planner_runner_, state.q);
        }
        if (sonic_planner_.should_replan(sample.command, now)) {
          sonic_planner_.replan(sonic_planner_runner_, sample.command, now);
          sonic_reference_cache_.replace(sonic_planner_.reference_frames());
          sonic_reference_frame_ = sonic_planner_.current_frame();
          sonic_realtime_heading_reset_ = true;
          event_log_.log("sonic_planner_replan",
                         {event_int("mode", sample.command.mode),
                          event_string("mode_name", sonic_planner_mode_name(sample.command.mode)),
                          event_number("speed", sample.command.speed),
                          event_int("frames", static_cast<long long>(sonic_planner_.size()))});
        }
        sonic_last_planner_sequence_ = sample.sequence;
      } else {
        const auto sample = sonic_zmq_.current_sample(sonic_stale_sec_);
        if (sample.sequence != sonic_last_realtime_sequence_ && !sample.frames.empty()) {
          sonic_reference_cache_.merge_tail(sample.frames);
          sonic_last_realtime_sequence_ = sample.sequence;
        }
      }
      sonic_hold_last_target_ = false;
      return true;
    }

    const auto sample = sonic_zmq_.current_sample(sonic_stale_sec_);
    const auto decision = sonic_stale_gate_.decide(sample.freshness, StaleBehavior::kHoldThenStop,
                                                   sonic_stale_stop_sec_, std::chrono::steady_clock::now());
    if (decision.new_stale_event) {
      ++sonic_stale_events_;
      event_log_.log("stale_event",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_string("source", "sonic_zmq_pose"),
                      event_int("step", policy_step)});
    }
    if (decision.action == StaleDecisionAction::kHoldLast) {
      ++sonic_stale_hold_ticks_;
      sonic_hold_last_target_ = true;
      return true;
    }
    if (decision.action == StaleDecisionAction::kStop) {
      ++sonic_stale_stop_events_;
      event_log_.log("stale_stop",
                     {event_string("player", player_label()), event_string("template", template_),
                      event_string("variant", variant_), event_string("source", "sonic_zmq_pose"),
                      event_int("step", policy_step)});
      throw std::runtime_error("SONIC realtime reference remained stale beyond the hold window");
    }
    if (sample.sequence != sonic_last_realtime_sequence_ && !sample.frames.empty()) {
      const int64_t incoming_step = sample.frames.size() > 1
          ? sample.frames[1].frame_index - sample.frames[0].frame_index
          : sonic_realtime_frame_step_;
      const bool compatible = !sonic_reference_cache_.empty() && incoming_step == sonic_realtime_frame_step_ &&
                              sample.frames.front().frame_index <=
                                  sonic_reference_cache_.last_frame_index() + sonic_realtime_frame_step_ &&
                              sample.frames.back().frame_index >= sonic_reference_frame_;
      if (compatible) {
        sonic_reference_cache_.merge_tail(sample.frames);
      } else {
        sonic_reference_cache_.replace(sample.frames);
        sonic_realtime_base_frame_ = sample.frames.front().frame_index;
        sonic_reference_frame_ = sonic_realtime_base_frame_;
        sonic_realtime_frame_step_ = incoming_step;
        sonic_realtime_heading_reset_ = true;
        event_log_.log("reference_window_reset",
                       {event_string("player", player_label()), event_string("variant", variant_),
                        event_int("first_frame", sonic_realtime_base_frame_),
                        event_int("frame_step", sonic_realtime_frame_step_)});
      }
      sonic_last_realtime_sequence_ = sample.sequence;
    }
    sonic_hold_last_target_ = false;
    return true;
  }

bool GenericPolicyPlayer::build_sonic_target(const GenericPolicyPlayer::RobotState& state, int64_t policy_step,
                          std::vector<float>& target, std::vector<float>& raw_action) {
    if (!update_sonic_reference(state, policy_step)) return false;
    if (sonic_hold_last_target_) return true;
    const auto robot = sonic_robot_frame(state);
    if (sonic_realtime_heading_reset_) {
      sonic_observation_builder_.reset_heading(robot, sonic_reference_cache_.frame_clamped(0));
      sonic_realtime_heading_reset_ = false;
    }
    sonic_observation_builder_.push_robot_frame(robot);
    if (sonic_input_source_ == "realtime-zmq" &&
        sonic_reference_frame_ > sonic_reference_cache_.last_frame_index()) {
      sonic_hold_last_target_ = true;
      return true;
    }
    const int64_t reference_index = sonic_input_source_ == "realtime-zmq"
                                        ? sonic_reference_cache_.local_index_for_frame(sonic_reference_frame_)
                                        : sonic_reference_frame_;
    const int64_t required_last = reference_index +
        static_cast<int64_t>(sonic_future_frames_ - 1) * sonic_future_step_;
    const int encoder_mode = sonic_reference_cache_.frame_clamped(reference_index).encoder_mode;
    if (sonic_input_source_ == "realtime-zmq" && encoder_mode == 0 &&
        required_last >= static_cast<int64_t>(sonic_reference_cache_.size())) {
      sonic_hold_last_target_ = true;
      return true;
    }
    const auto encoder_obs = sonic_observation_builder_.build_encoder_observation(
        sonic_reference_cache_, reference_index, sonic_future_frames_, sonic_future_step_);
    auto token = sonic_encoder_.run_single(encoder_obs, "obs_dict", "encoded_tokens", sonic_token_dim_);
    const auto decoder_obs = sonic_observation_builder_.build_decoder_observation(token);
    raw_action = sonic_decoder_.run_single(decoder_obs, "obs_dict", "action", motor_count_);
    last_action_ = raw_action;
    target = sonic_decode_target(raw_action, default_q_, action_scale_);
    for (int i = 0; i < motor_count_; ++i) {
      target[i] = std::clamp(target[i], lower_limits_[i], upper_limits_[i]);
    }
    if (sonic_input_source_ == "offline-csv") {
      if (sonic_reference_frame_ + 1 < static_cast<int64_t>(sonic_reference_cache_.size())) {
        ++sonic_reference_frame_;
      } else if (sonic_loop_reference_) {
        sonic_reference_frame_ = 0;
        sonic_observation_builder_.reset(robot, sonic_reference_cache_.frame_clamped(0));
      }
    } else {
      if (sonic_pico_control_ && sonic_use_planner_) {
        sonic_planner_.advance();
        sonic_reference_frame_ = sonic_planner_.current_frame();
      } else {
        sonic_reference_frame_ += sonic_realtime_frame_step_;
        const int64_t keep_from = sonic_reference_frame_ - 5 * sonic_realtime_frame_step_;
        sonic_reference_cache_.discard_before(keep_from);
      }
    }
    return true;
  }

void GenericPolicyPlayer::control_loop_sonic() {
    if (sonic_input_source_ == "realtime-zmq") {
      if (!wait_first_sonic_realtime()) {
        throw std::runtime_error(sonic_pico_control_
            ? "timeout waiting for PICO start command and fresh SONIC planner/pose packet"
            : "timeout waiting for first SONIC realtime ZMQ pose packet");
      }
      if (sonic_pico_control_) {
        if (!prepare_sonic_pico_source(snapshot(), true)) {
          throw std::runtime_error("PICO source became unavailable during planner initialization");
        }
      } else {
        const auto first = sonic_zmq_.current_sample(sonic_stale_sec_);
        if (first.frames.empty()) throw std::runtime_error("first SONIC realtime packet contains no frames");
        sonic_realtime_base_frame_ = first.frames.front().frame_index;
        sonic_reference_cache_.replace(first.frames);
        sonic_reference_frame_ = sonic_realtime_base_frame_;
        sonic_realtime_frame_step_ = first.frames.size() > 1
            ? first.frames[1].frame_index - first.frames[0].frame_index : 1;
        const int64_t required_frames = static_cast<int64_t>(sonic_future_frames_ - 1) * sonic_future_step_ + 1;
        if (first.frames.front().encoder_mode == 0 && static_cast<int64_t>(first.frames.size()) < required_frames) {
          throw std::runtime_error("first SONIC realtime packet does not contain the required lookahead window");
        }
        sonic_last_realtime_sequence_ = first.sequence;
      }
    }
    const int publish_hz = std::max(inference_hz_, publish_hz_);
    const auto policy_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, inference_hz_)));
    const auto publish_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / publish_hz));
    auto state = snapshot();
    reset_sonic_runtime(state);

    std::vector<float> hold_q, hold_kp, hold_kd;
    hold_current_pose(state, hold_q, hold_kp, hold_kd);
    publish(hold_q, hold_kp, hold_kd, state.mode_machine, false, CommandPhase::kHold);
    for (int i = 0; i < sonic_warmup_iters_; ++i) {
      std::vector<float> target, action;
      if (!build_sonic_target(state, -1, target, action) || target.empty() || action.empty()) {
        throw std::runtime_error("SONIC warmup did not produce a fresh target");
      }
    }
    reset_sonic_runtime(state);
    require_sonic_realtime_fresh("standby-ready", false);
    event_log_.log("standby_ready",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_string("source", sonic_input_source_)});
    std::cout << player_log_prefix() << " SONIC standby ready input=" << sonic_input_source_ << "\n";
    if (!ready_file_.empty()) touch_file(ready_file_);
    wait_start_file(hold_q, hold_kp, hold_kd, publish_hz, [this] {
      require_sonic_realtime_fresh("pre-handoff standby", true);
    }, true);
    if (g_stop_requested.load()) return;

    reset_sonic_runtime(snapshot());
    const auto ramp_start = std::chrono::steady_clock::now();
    while (!g_stop_requested.load()) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(now - ramp_start).count();
      if (elapsed >= sonic_startup_ramp_sec_) break;
      const float alpha = sonic_startup_ramp_sec_ <= 0.0 ? 1.f
          : static_cast<float>(std::clamp(elapsed / sonic_startup_ramp_sec_, 0.0, 1.0));
      state = snapshot();
      require_sonic_realtime_fresh("startup ramp", false);
      const auto ramp_q = sonic_startup_target(state.q, default_q_, alpha);
      publish(ramp_q, kp_, kd_, state.mode_machine);
      std::this_thread::sleep_for(publish_dt);
    }
    if (g_stop_requested.load()) return;

    state = snapshot();
    // Upstream starts its state logger on the first CONTROL tick. Clearing here
    // makes that tick one current frame with the older nine frames zero-padded.
    reset_sonic_runtime(state);
    std::vector<float> target = default_q_;
    CommandTiming target_timing;
    const auto start = std::chrono::steady_clock::now();
    std::mutex target_mutex;
    std::atomic<bool> worker_stop{false};
    std::atomic<bool> worker_done{false};
    std::exception_ptr worker_error;
    std::atomic<int64_t> inference_count{0}, publish_count{0};
    std::atomic<int64_t> policy_deadline_miss{0}, publish_deadline_miss{0};
    std::cout << player_log_prefix() << " start motion_conditioned_tracking/sonic provider="
              << sonic_decoder_.selected_execution_provider() << " input=" << sonic_input_source_
              << " hz=" << inference_hz_ << " publish_hz=" << publish_hz << "\n";
    std::thread inference_worker([&] {
      try {
        auto next_policy = start;
        std::vector<float> worker_target = target;
        while (!g_stop_requested.load() && !worker_stop.load()) {
          const auto now = std::chrono::steady_clock::now();
          if (now < next_policy) {
            std::this_thread::sleep_until(next_policy);
            continue;
          }
          policy_deadline_miss.fetch_add(advance_deadline(next_policy, policy_dt, now));
          const auto inference_state = snapshot();
          begin_command_timing(inference_state.stamp, inference_state.seq);
          std::vector<float> action;
          const int64_t step = inference_count.load();
          if (!build_sonic_target(inference_state, step, worker_target, action)) break;
          {
            std::lock_guard<std::mutex> lock(target_mutex);
            target = worker_target;
            target_timing = finish_command_timing();
          }
          const int64_t completed = inference_count.fetch_add(1) + 1;
          if (stats_interval_ > 0 && completed % stats_interval_ == 0 && !worker_target.empty()) {
            const auto range = std::minmax_element(worker_target.begin(), worker_target.end());
            event_log_.log("runtime_metrics",
                           {event_string("player", player_label()), event_string("template", template_),
                            event_string("variant", variant_), event_int("step", completed),
                            event_int("publish_count", active_publish_count_.load()),
                            event_int("policy_deadline_miss", policy_deadline_miss.load()),
                            event_int("publish_deadline_miss", publish_deadline_miss.load()),
                            event_number("max_publish_interval_ms",
                                         max_active_publish_interval_ms_.load()),
                            event_int("stale_events", sonic_stale_events_),
                            event_number("target_min", *range.first), event_number("target_max", *range.second)});
          }
          const bool offline_done = sonic_input_source_ == "offline-csv" && !sonic_loop_reference_ &&
                                    sonic_reference_frame_ >= static_cast<int64_t>(sonic_reference_cache_.size() - 1);
          if (offline_done || (max_frames_ > 0 && completed >= max_frames_) ||
              (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start).count() >= duration_sec_)) {
            break;
          }
        }
      } catch (...) {
        worker_error = std::current_exception();
      }
      worker_done.store(true);
    });

    auto next_publish = start;
    try {
      while (!g_stop_requested.load() && !worker_done.load()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_publish) {
          publish_deadline_miss.fetch_add(advance_deadline(next_publish, publish_dt, now));
          std::vector<float> publish_target;
          CommandTiming publish_timing;
          {
            std::lock_guard<std::mutex> lock(target_mutex);
            publish_target = target;
            publish_timing = target_timing;
          }
          const auto publish_state = snapshot();
          publish(publish_target, kp_, kd_, publish_state.mode_machine, false,
                  CommandPhase::kActive, publish_timing);
          publish_count.fetch_add(1);
        }
        if (next_publish > std::chrono::steady_clock::now()) std::this_thread::sleep_until(next_publish);
      }
    } catch (...) {
      worker_stop.store(true);
      if (inference_worker.joinable()) inference_worker.join();
      throw;
    }
    worker_stop.store(true);
    if (inference_worker.joinable()) inference_worker.join();
    if (worker_error) std::rethrow_exception(worker_error);
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    event_log_.log("runtime_metrics",
                   {event_string("player", player_label()), event_string("template", template_),
                    event_string("variant", variant_), event_int("step", inference_count.load()),
                    event_int("publish_count", publish_count.load()),
                    event_int("policy_deadline_miss", policy_deadline_miss.load()),
                    event_int("publish_deadline_miss", publish_deadline_miss.load()),
                    event_number("max_publish_interval_ms",
                                 max_active_publish_interval_ms_.load()),
                    event_int("stale_events", sonic_stale_events_),
                    event_int("stale_holds", sonic_stale_hold_ticks_),
                    event_int("stale_stops", sonic_stale_stop_events_), event_string("phase", "final")});
  }

std::array<float, 4> GenericPolicyPlayer::protomotions_torso_anchor_from_base_and_waist(const std::array<float, 4>& base_wxyz,
                                                                      const std::vector<float>& q) const {
    auto base = wxyz_to_xyzw(base_wxyz);
    auto yaw = axis_quat_xyzw(q.size() > 12 ? q[12] : 0.f, 'z');
    auto roll = axis_quat_xyzw(q.size() > 13 ? q[13] : 0.f, 'x');
    auto pitch = axis_quat_xyzw(q.size() > 14 ? q[14] : 0.f, 'y');
    return normalize_quat_xyzw(quat_mul_xyzw(base, quat_mul_xyzw(yaw, quat_mul_xyzw(roll, pitch))));
  }

std::array<float, 4> GenericPolicyPlayer::protomotions_anchor_quat(const GenericPolicyPlayer::RobotState& s) const {
    if (proto_anchor_source_ == "imu") return wxyz_to_xyzw(s.quat);
    return protomotions_torso_anchor_from_base_and_waist(s.quat, s.q);
  }

std::vector<float> GenericPolicyPlayer::protomotions_infer(const GenericPolicyPlayer::RobotState& s) {
    CommandTimingScope timing_scope(s.stamp, s.seq);
    if (s.q.size() < motor_count_ || s.dq.size() < motor_count_) {
      throw std::runtime_error("lowstate motor count too small");
    }
    if (proto_heading_uninitialized_) {
      auto motion_anchor = proto_motion_.body_rot_at(0, proto_anchor_idx_);
      proto_heading_offset_ = compute_yaw_offset_xyzw(protomotions_anchor_quat(s), motion_anchor);
      proto_heading_uninitialized_ = false;
      std::cout << player_log_prefix() << " protomotions heading offset initialized\n";
    }

    std::vector<float> future_anchor_rot(proto_future_steps_.size() * 4, 0.f);
    std::vector<float> future_dof_pos(proto_future_steps_.size() * motor_count_, 0.f);
    std::vector<float> future_dof_vel(proto_future_steps_.size() * motor_count_, 0.f);
    for (size_t i = 0; i < proto_future_steps_.size(); ++i) {
      const int f = std::clamp(proto_frame_ + proto_future_steps_[i], 0, proto_motion_.num_frames - 1);
      auto r = quat_mul_xyzw(proto_heading_offset_, proto_motion_.body_rot_at(f, proto_anchor_idx_));
      for (int j = 0; j < 4; ++j) future_anchor_rot[i * 4 + j] = r[j];
      std::vector<float> dp, dv;
      proto_motion_.dof_at(f, dp, dv);
      std::copy(dp.begin(), dp.end(), future_dof_pos.begin() + i * motor_count_);
      std::copy(dv.begin(), dv.end(), future_dof_vel.begin() + i * motor_count_);
    }

    std::unordered_map<std::string, OnnxRunner::TensorInput> key_data;
    key_data["current.dof_pos"] = {s.q, {1, motor_count_}};
    key_data["current.dof_vel"] = {s.dq, {1, motor_count_}};
    auto aq = protomotions_anchor_quat(s);
    key_data["current.anchor_rot"] = {{aq[0], aq[1], aq[2], aq[3]}, {1, 4}};
    key_data["current.root_local_ang_vel"] = {{s.gyro[0], s.gyro[1], s.gyro[2]}, {1, 3}};
    key_data["historical.processed_actions"] = {proto_prev_actions_, {1, 1, motor_count_}};
    key_data["mimic.future_anchor_rot"] = {future_anchor_rot, {1, static_cast<int64_t>(proto_future_steps_.size()), 4}};
    key_data["mimic.future_dof_pos"] = {future_dof_pos, {1, static_cast<int64_t>(proto_future_steps_.size()), motor_count_}};
    key_data["mimic.future_dof_vel"] = {future_dof_vel, {1, static_cast<int64_t>(proto_future_steps_.size()), motor_count_}};

    std::unordered_map<std::string, OnnxRunner::TensorInput> named_inputs;
    for (const auto& name : onnx_.input_names()) {
      const auto key_it = proto_name_to_key_.find(name);
      if (key_it == proto_name_to_key_.end()) throw std::runtime_error("no ProtoMotions semantic key for ONNX input: " + name);
      const auto data_it = key_data.find(key_it->second);
      if (data_it == key_data.end()) throw std::runtime_error("cannot build ProtoMotions ONNX input for key: " + key_it->second);
      named_inputs[name] = data_it->second;
    }

    auto outs = onnx_.run_named(named_inputs, {{"joint_pos_targets", motor_count_}});
    auto target = std::move(outs["joint_pos_targets"]);
    proto_prev_actions_ = target;
    return target;
  }

void GenericPolicyPlayer::protomotions_advance() {
    if (proto_frame_ < proto_motion_.num_frames - 1) {
      ++proto_frame_;
    } else if (proto_loop_motion_) {
      proto_frame_ = 0;
      proto_prev_actions_.assign(motor_count_, 0.f);
      proto_heading_uninitialized_ = true;
    }
  }

bool GenericPolicyPlayer::protomotions_done() const {
    return !proto_loop_motion_ && proto_frame_ >= proto_motion_.num_frames - 1;
  }

void GenericPolicyPlayer::control_loop_protomotions() {
    const int publish_hz = std::max(inference_hz_, publish_hz_);
    const auto policy_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, inference_hz_)));
    auto s = snapshot();
    std::vector<float> q, hold_kp, hold_kd;
    hold_current_pose(s, q, hold_kp, hold_kd);
    publish(q, hold_kp, hold_kd, s.mode_machine, false, CommandPhase::kHold);
    if (!ready_file_.empty()) {
      touch_file(ready_file_);
      std::cout << player_log_prefix() << " ready file touched: " << ready_file_ << "\n";
    }

    bool preview_committed = false;
    if (preview_mode_enabled()) {
      const int saved_frame = proto_frame_;
      const auto saved_prev_actions = proto_prev_actions_;
      const auto saved_heading_offset = proto_heading_offset_;
      const bool saved_heading_uninitialized = proto_heading_uninitialized_;
      auto restore_preview_state = [&]() {
        proto_frame_ = saved_frame;
        proto_prev_actions_ = saved_prev_actions;
        proto_heading_offset_ = saved_heading_offset;
        proto_heading_uninitialized_ = saved_heading_uninitialized;
      };

      q = protomotions_infer(s);
      protomotions_advance();
      const auto decision = wait_preview_decision(
          q, kp_, kd_, s.mode_machine, publish_hz, 0);
      if (decision != PreviewDecision::kCommit) {
        restore_preview_state();
        event_log_.log("preview_state_restored",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_)});
        if (decision == PreviewDecision::kTimeout) {
          throw std::runtime_error("transactional preview timed out before commit");
        }
        return;
      }
      preview_committed = true;
      event_log_.log("preview_native_startup_bypassed",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_number("configured_startup_interp_sec", proto_startup_interp_sec_)});
    } else {
      wait_start_file(q, hold_kp, hold_kd, publish_hz);
      if (g_stop_requested.load()) return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto next_policy = preview_committed ? start + policy_dt : start;
    const std::vector<float> startup_q = q;
    int64_t inference_count = 0;
    int64_t policy_deadline_miss = 0;
    bool max_frames_reached = false;
    AsyncCommandPublisher command_publisher(
        publish_hz, 0.0,
        JointCommand{q, preview_committed ? kp_ : hold_kp,
                     preview_committed ? kd_ : hold_kd},
        [this, start, startup_q, preview_committed](const JointCommand& command, bool active) {
          std::vector<float> q_cmd = command.q;
          if (active && !preview_committed && proto_startup_interp_sec_ > 1e-6 &&
              startup_q.size() == q_cmd.size()) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const float alpha = static_cast<float>(
                std::clamp(elapsed / proto_startup_interp_sec_, 0.0, 1.0));
            for (size_t i = 0; i < q_cmd.size(); ++i) {
              q_cmd[i] = (1.f - alpha) * startup_q[i] + alpha * q_cmd[i];
            }
          }
          const auto state = snapshot();
          publish(q_cmd, command.kp, command.kd, state.mode_machine, false,
                  active ? CommandPhase::kActive : CommandPhase::kHold, command.timing);
        },
        preview_committed);
    std::cout << player_log_prefix() << " start motion_conditioned_tracking/protomotions model="
              << runtime_model_path_.string()
              << " motion=" << trajectory_path_.string()
              << " frames=" << proto_motion_.num_frames
              << " hz=" << inference_hz_ << " publish_hz=" << publish_hz
              << " max_frames=" << max_frames_
              << " startup_interp_sec=" << proto_startup_interp_sec_ << "\n";
    std::exception_ptr inference_error;
    try {
      while (!g_stop_requested.load() && !command_publisher.failed()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_policy) {
          policy_deadline_miss += advance_deadline(next_policy, policy_dt, now);
          s = snapshot();
          q = protomotions_infer(s);
          protomotions_advance();
          command_publisher.update({q, kp_, kd_});
          ++inference_count;
          if (max_frames_ > 0 && inference_count >= max_frames_) max_frames_reached = true;
          if (stats_interval_ > 0 && inference_count % stats_interval_ == 0) {
            const auto mm = std::minmax_element(q.begin(), q.end());
            const auto metrics = command_publisher.metrics();
            std::cout << player_log_prefix() << " protomotions frame=" << proto_frame_
                      << "/" << (proto_motion_.num_frames - 1)
                      << " target=[" << *mm.first << "," << *mm.second << "]\n";
            event_log_.log("runtime_metrics",
                           {event_string("player", player_label()),
                            event_string("template", template_),
                            event_string("variant", variant_),
                            event_int("frame", proto_frame_),
                            event_int("step", inference_count),
                            event_int("publish_count", metrics.publish_count),
                            event_int("policy_deadline_miss", policy_deadline_miss),
                            event_int("publish_deadline_miss", metrics.deadline_miss),
                            event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                            event_number("target_min", *mm.first),
                            event_number("target_max", *mm.second)});
          }
        }
        if (max_frames_reached || protomotions_done()) break;
        if (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start).count() >= duration_sec_) break;
        if (next_policy > std::chrono::steady_clock::now()) std::this_thread::sleep_until(next_policy);
      }
    } catch (...) {
      inference_error = std::current_exception();
    }
    command_publisher.stop_and_join();
    command_publisher.rethrow_if_failed();
    if (inference_error) std::rethrow_exception(inference_error);
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    const auto metrics = command_publisher.metrics();
    std::cout << player_log_prefix() << " protomotions stats inference=" << inference_count
              << " publish=" << metrics.publish_count
              << " policy_deadline_miss=" << policy_deadline_miss
              << " publish_deadline_miss=" << metrics.deadline_miss << "\n";
    event_log_.log("runtime_metrics",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_int("step", inference_count),
                    event_int("publish_count", metrics.publish_count),
                    event_int("policy_deadline_miss", policy_deadline_miss),
                    event_int("publish_deadline_miss", metrics.deadline_miss),
                    event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                    event_string("phase", "final")});
    std::cout << player_log_prefix() << " completed protomotions\n";
  }

void GenericPolicyPlayer::reset_twist_controller() {
    twist_last_action_.assign(twist_policy_actions_, 0.f);
    twist_history_.clear();
    for (int i = 0; i < twist_history_len_; ++i) {
      twist_history_.push_back(std::vector<float>(twist_obs_frame_dim_, 0.f));
    }
    twist_stale_gate_.reset();
    twist_stale_events_ = 0;
    twist_stale_hold_ticks_ = 0;
    twist_stale_stop_events_ = 0;
  }

bool GenericPolicyPlayer::wait_first_twist_realtime() {
    if (twist_realtime_transport_ == "udp") return twist_udp_.wait_first(twist_input_timeout_sec_);
    if (twist_realtime_transport_ == "zmq") return twist_zmq_.wait_first(twist_input_timeout_sec_);
    if (twist_realtime_transport_ == "redis") return twist_redis_.wait_first(twist_input_timeout_sec_);
    return false;
  }

const char* GenericPolicyPlayer::twist_stale_behavior_name() const {
    if (twist_stale_behavior_ == StaleBehavior::kDefault) return "default";
    if (twist_stale_behavior_ == StaleBehavior::kStop) return "stop";
    return "hold_then_stop";
  }

TwistMimicSample GenericPolicyPlayer::current_twist_realtime_mimic() {
    if (twist_realtime_transport_ == "udp") return twist_udp_.current_sample(twist_stale_sec_);
    if (twist_realtime_transport_ == "zmq") return twist_zmq_.current_sample(twist_stale_sec_);
    if (twist_realtime_transport_ == "redis") return twist_redis_.current_sample(twist_stale_sec_);
    return {};
  }

TwistMimicSample GenericPolicyPlayer::current_twist_mimic(int64_t policy_step, bool& offline_finished) {
  offline_finished = false;
  if (twist_preview_sample_) return *twist_preview_sample_;
  if (twist_input_source_ == "realtime") return current_twist_realtime_mimic();
    const double t = static_cast<double>(policy_step) / static_cast<double>(std::max(1, inference_hz_));
    TwistMimicSample sample;
  sample.values = twist_offline_.frame_for_time(t, twist_loop_motion_, offline_finished);
  sample.freshness = offline_finished ? SampleFreshness::kMissing : SampleFreshness::kFresh;
  sample.sequence = static_cast<uint64_t>(std::max<int64_t>(0, policy_step) + 1);
  sample.age_sec = 0.0;
  sample.content_hash = float_content_hash(sample.values.data(), sample.values.size());
  return sample;
}

void GenericPolicyPlayer::split_twist_mimic(const std::array<float, kTwistProtocolMimicDim>& mimic,
                         std::vector<float>& policy_mimic,
                         std::array<float, 2>& wrist_roll) const {
    policy_mimic.clear();
    policy_mimic.reserve(static_cast<size_t>(twist_policy_mimic_dim_));
    wrist_roll = {mimic[27], mimic[32]};
    for (int i = 0; i < kTwistProtocolMimicDim; ++i) {
      if (i == 27 || i == 32) continue;
      policy_mimic.push_back(mimic[i]);
    }
  }

std::vector<float> GenericPolicyPlayer::build_twist_model_input(const GenericPolicyPlayer::RobotState& s, const std::array<float, kTwistProtocolMimicDim>& mimic) {
    if (s.q.size() < motor_count_ || s.dq.size() < motor_count_) throw std::runtime_error("lowstate motor count too small");
    if (twist_history_.size() != static_cast<size_t>(twist_history_len_)) reset_twist_controller();

    std::vector<float> policy_mimic;
    std::array<float, 2> wrist_roll{};
    split_twist_mimic(mimic, policy_mimic, wrist_roll);
    check_size("TWIST policy_mimic", static_cast<size_t>(twist_policy_mimic_dim_), policy_mimic.size());

    std::vector<float> dof_pos(twist_policy_actions_, 0.f), dof_vel(twist_policy_actions_, 0.f);
    for (int i = 0; i < twist_policy_actions_; ++i) {
      const int motor = twist_body_motor_idx_.at(i);
      dof_pos[i] = s.q.at(motor);
      dof_vel[i] = s.dq.at(motor);
    }
    for (int idx : twist_ankle_policy_idx_) {
      if (idx >= 0 && idx < twist_policy_actions_) dof_vel[idx] = 0.f;
    }
    const auto rpy = quat_rpy(s.quat);

    std::vector<float> obs_proprio;
    obs_proprio.reserve(static_cast<size_t>(twist_proprio_dim_));
    for (float g : s.gyro) obs_proprio.push_back(g * twist_ang_vel_scale_);
    obs_proprio.push_back(rpy[0]);
    obs_proprio.push_back(rpy[1]);
    for (int i = 0; i < twist_policy_actions_; ++i) {
      obs_proprio.push_back((dof_pos[i] - twist_default_dof_pos_[i]) * twist_dof_pos_scale_);
    }
    for (int i = 0; i < twist_policy_actions_; ++i) obs_proprio.push_back(dof_vel[i] * twist_dof_vel_scale_);
    obs_proprio.insert(obs_proprio.end(), twist_last_action_.begin(), twist_last_action_.end());
    check_size("TWIST obs_proprio", static_cast<size_t>(twist_proprio_dim_), obs_proprio.size());

    std::vector<float> obs_frame;
    obs_frame.reserve(static_cast<size_t>(twist_obs_frame_dim_));
    obs_frame.insert(obs_frame.end(), policy_mimic.begin(), policy_mimic.end());
    obs_frame.insert(obs_frame.end(), obs_proprio.begin(), obs_proprio.end());
    check_size("TWIST obs_frame", static_cast<size_t>(twist_obs_frame_dim_), obs_frame.size());

    std::vector<float> input;
    input.reserve(static_cast<size_t>(twist_input_dim_));
    input.insert(input.end(), obs_frame.begin(), obs_frame.end());
    for (const auto& h : twist_history_) input.insert(input.end(), h.begin(), h.end());
    check_size("TWIST model_input", static_cast<size_t>(twist_input_dim_), input.size());

    twist_history_.pop_front();
    twist_history_.push_back(std::move(obs_frame));
    return input;
  }

std::vector<float> GenericPolicyPlayer::twist_infer(const std::vector<float>& obs) {
    check_size("TWIST obs", static_cast<size_t>(twist_obs_dim_), obs.size());
    return onnx_.run_single(obs, obs_input_name_, action_output_name_, twist_policy_actions_);
  }

GenericPolicyPlayer::TwistTargetStatus GenericPolicyPlayer::build_twist_target(const GenericPolicyPlayer::RobotState& s, int64_t policy_step,
                                       std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                                       std::vector<float>& raw_action) {
    CommandTimingScope timing_scope(s.stamp, s.seq);
    bool offline_finished = false;
    auto sample = current_twist_mimic(policy_step, offline_finished);
    if (offline_finished && twist_input_source_ != "realtime") return TwistTargetStatus::kStop;
    if (twist_input_source_ == "realtime") {
      const auto decision = twist_stale_gate_.decide(sample.freshness, twist_stale_behavior_,
                                                     twist_input_timeout_sec_, std::chrono::steady_clock::now());
      if (decision.new_stale_event) {
        ++twist_stale_events_;
        const char* status = sample.freshness == SampleFreshness::kMissing ? "missing" : "stale";
        std::cerr << player_log_prefix() << " TWIST realtime input " << status
                  << "; behavior=" << twist_stale_behavior_name()
                  << " timeout=" << twist_input_timeout_sec_ << "s\n";
        event_log_.log("stale_event",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_string("source", "twist_realtime"),
                        event_string("freshness", status),
                        event_string("behavior", twist_stale_behavior_name()),
                        event_int("step", policy_step)});
      }
      if (decision.action == StaleDecisionAction::kUseDefault) {
        sample.values = twist_default_mimic_obs_;
      } else if (decision.action == StaleDecisionAction::kHoldLast) {
        ++twist_stale_hold_ticks_;
        return TwistTargetStatus::kHoldLast;
      } else if (decision.action == StaleDecisionAction::kStop) {
        ++twist_stale_stop_events_;
        std::cerr << player_log_prefix() << " TWIST realtime input stale timeout; stopping action after "
                  << decision.hold_elapsed_sec << "s\n";
        return TwistTargetStatus::kStop;
      }
    }

    std::vector<float> policy_mimic;
    std::array<float, 2> wrist_roll{};
    split_twist_mimic(sample.values, policy_mimic, wrist_roll);

    auto model_input = build_twist_model_input(s, sample.values);
    raw_action = twist_infer(model_input);
    twist_last_action_ = raw_action;

    q.assign(motor_count_, 0.f);
    kp.assign(motor_count_, 0.f);
    kd.assign(motor_count_, 0.f);
    for (int i = 0; i < twist_policy_actions_; ++i) {
      const int motor = twist_body_motor_idx_.at(i);
      const float clipped = std::clamp(raw_action[i], -10.f, 10.f);
      q[motor] = twist_default_dof_pos_[i] + clipped * twist_action_scale_;
      kp[motor] = twist_body_kp_[i];
      kd[motor] = twist_body_kd_[i];
    }
    for (int i = 0; i < 6; ++i) {
      const int motor = twist_wrist_motor_idx_.at(i);
      q[motor] = twist_wrist_default_[i];
      kp[motor] = twist_wrist_kp_[i];
      kd[motor] = twist_wrist_kd_[i];
    }
    q[19] = wrist_roll[0];
    q[26] = wrist_roll[1];
    return TwistTargetStatus::kCommand;
  }

void GenericPolicyPlayer::control_loop_twist() {
    if (twist_input_source_ == "realtime" && twist_require_realtime_ && !wait_first_twist_realtime()) {
      throw std::runtime_error("timeout waiting for first TWIST realtime mimic_obs packet");
    }
    reset_twist_controller();

    const int publish_hz = std::max(inference_hz_, publish_hz_);
    const auto policy_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, inference_hz_)));
    auto s = snapshot();
    std::vector<float> q, kp, kd;
    hold_current_pose(s, q, kp, kd);
    publish(q, kp, kd, s.mode_machine, false, CommandPhase::kHold);
    if (!ready_file_.empty()) {
      touch_file(ready_file_);
      std::cout << player_log_prefix() << " ready file touched: " << ready_file_ << "\n";
    }

    bool preview_committed = false;
    int64_t initial_policy_step = 0;
    std::vector<float> preview_raw_action(twist_policy_actions_, 0.f);
    if (preview_mode_enabled()) {
      const auto saved_last_action = twist_last_action_;
      const auto saved_history = twist_history_;
      const auto saved_stale_gate = twist_stale_gate_;
      const auto saved_stale_events = twist_stale_events_;
      const auto saved_stale_holds = twist_stale_hold_ticks_;
      const auto saved_stale_stops = twist_stale_stop_events_;

      bool offline_finished = false;
      TwistMimicSample frozen_input;
      if (twist_input_source_ == "realtime") {
        frozen_input = current_twist_realtime_mimic();
      } else {
        frozen_input.values = twist_offline_.frame_for_time(0.0, twist_loop_motion_, offline_finished);
        frozen_input.freshness = offline_finished ? SampleFreshness::kMissing : SampleFreshness::kFresh;
        frozen_input.sequence = 1;
        frozen_input.age_sec = 0.0;
        frozen_input.content_hash = float_content_hash(
            frozen_input.values.data(), frozen_input.values.size());
      }
      twist_preview_sample_ = frozen_input;
      event_log_.log("preview_input_snapshot",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_string("source", "twist_motion_packet"),
                      event_int("input_sequence", static_cast<long long>(frozen_input.sequence)),
                      event_number("input_freshness_sec", frozen_input.age_sec),
                      event_string("input_hash", frozen_input.content_hash)});

      std::vector<float> candidate_q, candidate_kp, candidate_kd;
      const auto status = build_twist_target(
          s, 0, candidate_q, candidate_kp, candidate_kd, preview_raw_action);
      if (status != TwistTargetStatus::kCommand) {
        twist_preview_sample_.reset();
        throw std::runtime_error("TWIST preview requires one fresh, usable input snapshot");
      }
      q = std::move(candidate_q);
      kp = std::move(candidate_kp);
      kd = std::move(candidate_kd);
      const auto decision = wait_preview_decision(q, kp, kd, s.mode_machine, publish_hz, 0);
      twist_preview_sample_.reset();
      if (decision != PreviewDecision::kCommit) {
        twist_last_action_ = saved_last_action;
        twist_history_ = saved_history;
        twist_stale_gate_ = saved_stale_gate;
        twist_stale_events_ = saved_stale_events;
        twist_stale_hold_ticks_ = saved_stale_holds;
        twist_stale_stop_events_ = saved_stale_stops;
        event_log_.log("preview_state_restored",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_)});
        if (decision == PreviewDecision::kTimeout) {
          throw std::runtime_error("transactional preview timed out before commit");
        }
        return;
      }
      preview_committed = true;
      initial_policy_step = 1;
    } else {
      wait_start_file(q, kp, kd, publish_hz);
      if (g_stop_requested.load()) return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto next_policy = preview_committed ? start + policy_dt : start;
    int64_t policy_step = initial_policy_step;
    int64_t policy_deadline_miss = 0;
    bool finished = false;
    std::vector<float> raw_action = preview_committed
        ? preview_raw_action : std::vector<float>(twist_policy_actions_, 0.f);
    AsyncCommandPublisher command_publisher(
        publish_hz, 0.0, JointCommand{q, kp, kd},
        [this](const JointCommand& command, bool active) {
          const auto state = snapshot();
          publish(command.q, command.kp, command.kd, state.mode_machine, false,
                  active ? CommandPhase::kActive : CommandPhase::kHold, command.timing);
        },
        preview_committed);

    std::cout << player_log_prefix() << " start motion_conditioned_tracking/twist model="
              << runtime_model_path_.string()
              << " input=" << twist_input_source_
              << " transport=" << twist_realtime_transport_
              << " hz=" << inference_hz_
              << " publish_hz=" << publish_hz
              << " action_scale=" << twist_action_scale_
              << " stale_behavior=" << twist_stale_behavior_name()
              << " max_frames=" << max_frames_ << "\n";
    std::exception_ptr inference_error;
    try {
      while (!g_stop_requested.load() && !command_publisher.failed()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_policy) {
          policy_deadline_miss += advance_deadline(next_policy, policy_dt, now);
          s = snapshot();
          std::vector<float> next_q, next_kp, next_kd, next_action;
          const auto status = build_twist_target(s, policy_step, next_q, next_kp, next_kd, next_action);
          if (status == TwistTargetStatus::kStop) {
            finished = true;
          } else if (status == TwistTargetStatus::kCommand) {
            q = std::move(next_q);
            kp = std::move(next_kp);
            kd = std::move(next_kd);
            raw_action = std::move(next_action);
            command_publisher.update({q, kp, kd});
          }
          ++policy_step;
          if (stats_interval_ > 0 && policy_step % stats_interval_ == 0) {
            const auto mm = std::minmax_element(raw_action.begin(), raw_action.end());
            const auto metrics = command_publisher.metrics();
            std::cout << player_log_prefix() << " twist step=" << policy_step
                      << " action=[" << *mm.first << "," << *mm.second << "]"
                      << " stale_events=" << twist_stale_events_
                      << " stale_holds=" << twist_stale_hold_ticks_
                      << " stale_stops=" << twist_stale_stop_events_ << "\n";
            event_log_.log("runtime_metrics",
                           {event_string("player", player_label()),
                            event_string("template", template_),
                            event_string("variant", variant_),
                            event_int("step", policy_step),
                            event_int("publish_count", metrics.publish_count),
                            event_int("policy_deadline_miss", policy_deadline_miss),
                            event_int("publish_deadline_miss", metrics.deadline_miss),
                            event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                            event_int("stale_events", twist_stale_events_),
                            event_int("stale_holds", twist_stale_hold_ticks_),
                            event_int("stale_stops", twist_stale_stop_events_),
                            event_number("action_min", *mm.first),
                            event_number("action_max", *mm.second)});
          }
        }
        if (finished) break;
        if (max_frames_ > 0 && policy_step >= max_frames_) break;
        if (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start).count() >= duration_sec_) break;
        if (next_policy > std::chrono::steady_clock::now()) std::this_thread::sleep_until(next_policy);
      }
    } catch (...) {
      inference_error = std::current_exception();
    }
    command_publisher.stop_and_join();
    command_publisher.rethrow_if_failed();
    if (inference_error) std::rethrow_exception(inference_error);
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    const auto metrics = command_publisher.metrics();
    std::cout << player_log_prefix() << " twist stats policy_steps=" << policy_step
              << " publish=" << metrics.publish_count
              << " policy_deadline_miss=" << policy_deadline_miss
              << " publish_deadline_miss=" << metrics.deadline_miss
              << " stale_events=" << twist_stale_events_
              << " stale_holds=" << twist_stale_hold_ticks_
              << " stale_stops=" << twist_stale_stop_events_ << "\n";
    event_log_.log("runtime_metrics",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_int("step", policy_step),
                    event_int("publish_count", metrics.publish_count),
                    event_int("policy_deadline_miss", policy_deadline_miss),
                    event_int("publish_deadline_miss", metrics.deadline_miss),
                    event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                    event_int("stale_events", twist_stale_events_),
                    event_int("stale_holds", twist_stale_hold_ticks_),
                    event_int("stale_stops", twist_stale_stop_events_),
                    event_string("phase", "final")});
    std::cout << player_log_prefix() << " completed twist\n";
  }

HumanoidGptReferenceFrame GenericPolicyPlayer::default_humanoid_gpt_reference() const {
    HumanoidGptReferenceFrame ref;
    ref.qpos.fill(0.f);
    ref.qpos[2] = 0.78f;
    ref.qpos[3] = 1.f;
    for (int i = 0; i < motor_count_ && i < static_cast<int>(default_q_.size()); ++i) {
      ref.qpos[7 + i] = default_q_[i];
    }
    ref.qvel.fill(0.f);
    ref.gv2wrd_pose = navigation_pose_from_qpos(ref.qpos);
    for (int k = 0; k < kHumanoidGptProtocolKeypointCount; ++k) {
      auto T = identity_pose4();
      T[11] = ref.qpos[2];
      std::copy(T.begin(), T.end(), ref.kpt2gv_pose.begin() + k * 16);
    }
    ref.kpt_cvel_in_gv.fill(0.f);
    ref.gv_vel = {0.f, 0.f, 0.f};
    return ref;
  }

bool GenericPolicyPlayer::wait_first_humanoid_gpt_realtime() {
    if (humanoid_gpt_realtime_transport_ == "udp") return humanoid_gpt_udp_.wait_first(humanoid_gpt_input_timeout_sec_);
    return false;
  }

const char* GenericPolicyPlayer::humanoid_gpt_stale_behavior_name() const {
    if (humanoid_gpt_stale_behavior_ == StaleBehavior::kDefault) return "default";
    if (humanoid_gpt_stale_behavior_ == StaleBehavior::kStop) return "stop";
    return "hold_then_stop";
  }

bool GenericPolicyPlayer::humanoid_gpt_offline_done() const {
    return !humanoid_gpt_loop_reference_ && humanoid_gpt_ref_frame_ >= humanoid_gpt_reference_cache_.num_frames() - 1;
  }

GenericPolicyPlayer::HumanoidGptTargetStatus GenericPolicyPlayer::update_humanoid_gpt_reference(const GenericPolicyPlayer::RobotState& s, int64_t policy_step) {
    if (humanoid_gpt_input_source_ == "offline-npz") {
      if (humanoid_gpt_reference_cache_.num_frames() <= 0) return HumanoidGptTargetStatus::kStop;
      const int f = std::clamp(humanoid_gpt_ref_frame_, 0, humanoid_gpt_reference_cache_.num_frames() - 1);
      const int n = std::min(f + 1, humanoid_gpt_reference_cache_.num_frames() - 1);
      humanoid_gpt_ref_curr_ = humanoid_gpt_reference_cache_.frame_at(f);
      humanoid_gpt_ref_next_ = humanoid_gpt_reference_cache_.frame_at(n);
      if (humanoid_gpt_ref_frame_ < humanoid_gpt_reference_cache_.num_frames() - 1) {
        ++humanoid_gpt_ref_frame_;
      } else if (humanoid_gpt_loop_reference_) {
        humanoid_gpt_ref_frame_ = 0;
        last_action_.assign(motor_count_, 0.f);
      } else {
        return policy_step == 0 ? HumanoidGptTargetStatus::kCommand : HumanoidGptTargetStatus::kStop;
      }
      return HumanoidGptTargetStatus::kCommand;
    }

    auto sample = humanoid_gpt_udp_.current_sample(humanoid_gpt_stale_sec_);
    const auto decision = humanoid_gpt_stale_gate_.decide(sample.freshness, humanoid_gpt_stale_behavior_,
                                                          humanoid_gpt_input_timeout_sec_, std::chrono::steady_clock::now());
    if (decision.new_stale_event) {
      ++humanoid_gpt_stale_events_;
      const char* status = sample.freshness == SampleFreshness::kMissing ? "missing" : "stale";
      std::cerr << player_log_prefix() << " Humanoid-GPT realtime input " << status
                << "; behavior=" << humanoid_gpt_stale_behavior_name()
                << " timeout=" << humanoid_gpt_input_timeout_sec_ << "s\n";
      event_log_.log("stale_event",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_string("source", "humanoid_gpt_realtime"),
                      event_string("freshness", status),
                      event_string("behavior", humanoid_gpt_stale_behavior_name()),
                      event_int("step", policy_step)});
    }
    if (decision.action == StaleDecisionAction::kUseDefault) {
      humanoid_gpt_ref_curr_ = default_humanoid_gpt_reference();
      humanoid_gpt_ref_next_ = humanoid_gpt_ref_curr_;
      return HumanoidGptTargetStatus::kCommand;
    }
    if (decision.action == StaleDecisionAction::kHoldLast) {
      ++humanoid_gpt_stale_hold_ticks_;
      return HumanoidGptTargetStatus::kHoldLast;
    }
    if (decision.action == StaleDecisionAction::kStop) {
      ++humanoid_gpt_stale_stop_events_;
      std::cerr << player_log_prefix() << " Humanoid-GPT realtime input stale timeout; stopping action after "
                << decision.hold_elapsed_sec << "s\n";
      return HumanoidGptTargetStatus::kStop;
    }
    if (!humanoid_gpt_converter_.loaded()) {
      throw std::runtime_error("Humanoid-GPT realtime converter is not loaded");
    }
    const auto converted = humanoid_gpt_converter_.convert(sample.qpos);
    if (!humanoid_gpt_have_realtime_ref_) {
      humanoid_gpt_ref_curr_ = converted;
      humanoid_gpt_ref_next_ = converted;
      humanoid_gpt_have_realtime_ref_ = true;
    } else {
      humanoid_gpt_ref_curr_ = humanoid_gpt_ref_next_;
      humanoid_gpt_ref_next_ = converted;
    }
    return HumanoidGptTargetStatus::kCommand;
  }

std::vector<float> GenericPolicyPlayer::build_humanoid_gpt_obs(const GenericPolicyPlayer::RobotState& s,
                                            const HumanoidGptReferenceFrame& ref_curr,
                                            const HumanoidGptReferenceFrame& ref_next) const {
    if (s.q.size() < motor_count_ || s.dq.size() < motor_count_) {
      throw std::runtime_error("lowstate motor count too small");
    }
    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(humanoid_gpt_obs_dim_));
    for (float v : s.gyro) obs.push_back(v);
    const auto g = projected_gravity_from_quat(s.quat);
    for (float v : g) obs.push_back(v);
    for (int i = 0; i < motor_count_; ++i) obs.push_back(s.q[i] - default_q_[i]);
    for (int i = 0; i < motor_count_; ++i) obs.push_back(s.dq[i]);
    for (int i = 0; i < motor_count_; ++i) obs.push_back(last_action_[i]);
    for (int i = 0; i < motor_count_; ++i) obs.push_back(ref_next.qpos[7 + i] - default_q_[i]);

    const size_t root = 0;
    obs.push_back(ref_next.kpt2gv_pose[root * 16 + 11]);
    obs.push_back(-ref_next.kpt2gv_pose[root * 16 + 8]);
    obs.push_back(-ref_next.kpt2gv_pose[root * 16 + 9]);
    obs.push_back(-ref_next.kpt2gv_pose[root * 16 + 10]);
    for (int i = 0; i < 6; ++i) obs.push_back(ref_next.kpt_cvel_in_gv[root * 6 + i]);

    const float yaw_ref = quat_yaw({ref_curr.qpos[3], ref_curr.qpos[4], ref_curr.qpos[5], ref_curr.qpos[6]});
    const float yaw_curr = quat_yaw(s.quat);
    const float yaw_d = wrap_pi(yaw_ref - yaw_curr);
    obs.push_back(std::cos(yaw_d));
    obs.push_back(std::sin(yaw_d));

    const float dx = ref_curr.qpos[0];
    const float dy = ref_curr.qpos[1];
    const float c = std::cos(-yaw_curr);
    const float ss = std::sin(-yaw_curr);
    obs.push_back(c * dx - ss * dy);
    obs.push_back(ss * dx + c * dy);
    check_size("Humanoid-GPT obs", static_cast<size_t>(humanoid_gpt_obs_dim_), obs.size());
    return obs;
  }

std::vector<float> GenericPolicyPlayer::humanoid_gpt_infer(const std::vector<float>& obs) {
    check_size("Humanoid-GPT obs", static_cast<size_t>(humanoid_gpt_obs_dim_), obs.size());
    return onnx_.run_single(obs, obs_input_name_, action_output_name_, motor_count_);
  }

GenericPolicyPlayer::HumanoidGptTargetStatus GenericPolicyPlayer::build_humanoid_gpt_target(const GenericPolicyPlayer::RobotState& s, int64_t policy_step,
                                                    std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                                                    std::vector<float>& raw_action) {
    CommandTimingScope timing_scope(s.stamp, s.seq);
    const auto ref_status = update_humanoid_gpt_reference(s, policy_step);
    if (ref_status != HumanoidGptTargetStatus::kCommand) return ref_status;
    const auto obs = build_humanoid_gpt_obs(s, humanoid_gpt_ref_curr_, humanoid_gpt_ref_next_);
    raw_action = humanoid_gpt_infer(obs);
    last_action_ = raw_action;
    q.assign(motor_count_, 0.f);
    kp = kp_;
    kd = kd_;
    for (int i = 0; i < motor_count_; ++i) {
      q[i] = default_q_[i] + raw_action[i] * action_scale_[i];
      q[i] = std::clamp(q[i], lower_limits_[i], upper_limits_[i]);
    }
    return HumanoidGptTargetStatus::kCommand;
  }

void GenericPolicyPlayer::reset_humanoid_gpt_runtime(const GenericPolicyPlayer::RobotState& s) {
    last_action_.assign(motor_count_, 0.f);
    humanoid_gpt_stale_gate_.reset();
    humanoid_gpt_stale_events_ = 0;
    humanoid_gpt_stale_hold_ticks_ = 0;
    humanoid_gpt_stale_stop_events_ = 0;
    humanoid_gpt_have_realtime_ref_ = false;
    if (humanoid_gpt_input_source_ == "realtime") {
      humanoid_gpt_converter_.reset();
      humanoid_gpt_converter_.set_robot_initial_pose(s.quat, {0.f, 0.f});
    }
  }

void GenericPolicyPlayer::control_loop_humanoid_gpt() {
    if (humanoid_gpt_input_source_ == "realtime" && humanoid_gpt_require_realtime_ && !wait_first_humanoid_gpt_realtime()) {
      throw std::runtime_error("timeout waiting for first Humanoid-GPT HMCP qpos36 packet");
    }

    const int publish_hz = std::max(inference_hz_, publish_hz_);
    const auto policy_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, inference_hz_)));
    auto s = snapshot();
    std::vector<float> q, kp, kd;
    hold_current_pose(s, q, kp, kd);
    publish(q, kp, kd, s.mode_machine, false, CommandPhase::kHold);
    if (!ready_file_.empty()) {
      touch_file(ready_file_);
      std::cout << player_log_prefix() << " ready file touched: " << ready_file_ << "\n";
    }
    wait_start_file(q, kp, kd, publish_hz);
    if (g_stop_requested.load()) return;
    s = snapshot();
    reset_humanoid_gpt_runtime(s);

    const auto start = std::chrono::steady_clock::now();
    auto next_policy = start;
    int64_t policy_step = 0;
    int64_t policy_deadline_miss = 0;
    bool finished = false;
    std::vector<float> raw_action(motor_count_, 0.f);
    AsyncCommandPublisher command_publisher(
        publish_hz, 0.0, JointCommand{q, kp, kd},
        [this](const JointCommand& command, bool active) {
          const auto state = snapshot();
          publish(command.q, command.kp, command.kd, state.mode_machine, false,
                  active ? CommandPhase::kActive : CommandPhase::kHold, command.timing);
        });

    std::cout << player_log_prefix() << " start motion_conditioned_tracking/humanoid_gpt model="
              << runtime_model_path_.string()
              << " input=" << humanoid_gpt_input_source_
              << " transport=" << humanoid_gpt_realtime_transport_
              << " hz=" << inference_hz_
              << " publish_hz=" << publish_hz
              << " stale_behavior=" << humanoid_gpt_stale_behavior_name()
              << " max_frames=" << max_frames_ << "\n";
    std::exception_ptr inference_error;
    try {
      while (!g_stop_requested.load() && !command_publisher.failed()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_policy) {
          policy_deadline_miss += advance_deadline(next_policy, policy_dt, now);
          s = snapshot();
          std::vector<float> next_q, next_kp, next_kd, next_action;
          const auto status = build_humanoid_gpt_target(s, policy_step, next_q, next_kp, next_kd, next_action);
          if (status == HumanoidGptTargetStatus::kStop) {
            finished = true;
          } else if (status == HumanoidGptTargetStatus::kCommand) {
            q = std::move(next_q);
            kp = std::move(next_kp);
            kd = std::move(next_kd);
            raw_action = std::move(next_action);
            command_publisher.update({q, kp, kd});
          }
          ++policy_step;
          if (stats_interval_ > 0 && policy_step % stats_interval_ == 0) {
            const auto am = std::minmax_element(raw_action.begin(), raw_action.end());
            const auto qm = std::minmax_element(q.begin(), q.end());
            const auto metrics = command_publisher.metrics();
            std::cout << player_log_prefix() << " humanoid_gpt step=" << policy_step
                      << " action=[" << *am.first << "," << *am.second << "]"
                      << " target_q=[" << *qm.first << "," << *qm.second << "]"
                      << " stale_events=" << humanoid_gpt_stale_events_
                      << " stale_holds=" << humanoid_gpt_stale_hold_ticks_
                      << " stale_stops=" << humanoid_gpt_stale_stop_events_ << "\n";
            event_log_.log("runtime_metrics",
                           {event_string("player", player_label()),
                            event_string("template", template_),
                            event_string("variant", variant_),
                            event_int("step", policy_step),
                            event_int("publish_count", metrics.publish_count),
                            event_int("policy_deadline_miss", policy_deadline_miss),
                            event_int("publish_deadline_miss", metrics.deadline_miss),
                            event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                            event_int("stale_events", humanoid_gpt_stale_events_),
                            event_int("stale_holds", humanoid_gpt_stale_hold_ticks_),
                            event_int("stale_stops", humanoid_gpt_stale_stop_events_),
                            event_number("action_min", *am.first),
                            event_number("action_max", *am.second),
                            event_number("target_min", *qm.first),
                            event_number("target_max", *qm.second)});
          }
        }
        if (finished) break;
        if (max_frames_ > 0 && policy_step >= max_frames_) break;
        if (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start).count() >= duration_sec_) break;
        if (humanoid_gpt_input_source_ == "offline-npz" && humanoid_gpt_offline_done()) break;
        if (next_policy > std::chrono::steady_clock::now()) std::this_thread::sleep_until(next_policy);
      }
    } catch (...) {
      inference_error = std::current_exception();
    }
    command_publisher.stop_and_join();
    command_publisher.rethrow_if_failed();
    if (inference_error) std::rethrow_exception(inference_error);
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    const auto metrics = command_publisher.metrics();
    std::cout << player_log_prefix() << " humanoid_gpt stats policy_steps=" << policy_step
              << " publish=" << metrics.publish_count
              << " policy_deadline_miss=" << policy_deadline_miss
              << " publish_deadline_miss=" << metrics.deadline_miss
              << " stale_events=" << humanoid_gpt_stale_events_
              << " stale_holds=" << humanoid_gpt_stale_hold_ticks_
              << " stale_stops=" << humanoid_gpt_stale_stop_events_ << "\n";
    event_log_.log("runtime_metrics",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_int("step", policy_step),
                    event_int("publish_count", metrics.publish_count),
                    event_int("policy_deadline_miss", policy_deadline_miss),
                    event_int("publish_deadline_miss", metrics.deadline_miss),
                    event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                    event_int("stale_events", humanoid_gpt_stale_events_),
                    event_int("stale_holds", humanoid_gpt_stale_hold_ticks_),
                    event_int("stale_stops", humanoid_gpt_stale_stop_events_),
                    event_string("phase", "final")});
    std::cout << player_log_prefix() << " completed humanoid_gpt\n";
  }

}  // namespace action_go::runtime::generic_policy_runtime
