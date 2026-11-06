#include "generic_policy_runtime_internal.h"

namespace action_go::runtime::generic_policy_runtime {

namespace {

int required_joint_index(const std::vector<std::string>& joint_order, const std::string& joint_name) {
    const auto it = std::find(joint_order.begin(), joint_order.end(), joint_name);
    if (it == joint_order.end()) {
      throw std::runtime_error("policy joint order is missing required torso joint: " + joint_name);
    }
    return static_cast<int>(std::distance(joint_order.begin(), it));
}

}  // namespace

void GenericPolicyPlayer::load_spec_beyondmimic(const YAML::Node& y) {
    const YAML::Node mimic = y["mimic_time"];
    beyond_ref_body_count_ = yamlc::required_scalar<int>(y, "mimic_time.ref_body_count");
    beyond_ref_anchor_idx_ = yamlc::required_scalar<int>(y, "mimic_time.anchor_body_index");
    if (beyond_ref_body_count_ != protocol_->ref_body_count ||
        beyond_ref_anchor_idx_ != protocol_->ref_anchor_index) {
      throw std::runtime_error("BeyondMimic mimic_time ref_body_count/anchor_body_index do not match protocol constraint");
    }
    beyond_warmup_steps_ = int_or(mimic, "warmup_steps", beyond_warmup_steps_);
    beyond_initial_reference_step_ = int_or(mimic, "initial_reference_step", 0);
    if (beyond_initial_reference_step_ < 0 || beyond_initial_reference_step_ >= max_frames_) {
      throw std::runtime_error("mimic_time.initial_reference_step must be in [0, max_frames)");
    }
    const std::string yaw_alignment = scalar_or<std::string>(mimic, "yaw_alignment", "anchor_yaw");
    if (yaw_alignment == "anchor_yaw") {
      beyond_align_initial_yaw_ = true;
    } else if (yaw_alignment == "none") {
      beyond_align_initial_yaw_ = false;
    } else {
      throw std::runtime_error("mimic_time.yaw_alignment must be anchor_yaw or none");
    }
    const std::string policy_step_source =
        scalar_or<std::string>(mimic, "policy_step_source", "wall_clock");
    if (policy_step_source == "lowstate_tick") {
      beyond_use_lowstate_tick_ = true;
    } else if (policy_step_source == "wall_clock") {
      beyond_use_lowstate_tick_ = false;
    } else {
      throw std::runtime_error("mimic_time.policy_step_source must be wall_clock or lowstate_tick");
    }
    beyond_policy_tick_ms_ = static_cast<uint32_t>(
        int_or(mimic, "policy_tick_ms", std::max(1, 1000 / std::max(1, inference_hz_))));
    beyond_sim_lockstep_sequence_ =
        scalar_or<bool>(mimic, "sim_lockstep_sequence", false);
  }

void GenericPolicyPlayer::load_spec_holosoma(const YAML::Node& y) {
    const YAML::Node mimic = y["mimic_time"];
    holosoma_motion_start_step_ = scalar_or<int>(mimic, "motion_start_step", 0);
    holosoma_motion_end_step_ = args_.max_frames >= 0 ? args_.max_frames : scalar_or<int>(mimic, "motion_end_step", max_frames_);
    holosoma_prepare_target_ = scalar_or<std::string>(mimic, "prepare_target", "motion");
    holosoma_prepare_ramp_sec_ = scalar_or<double>(mimic, "prepare_ramp_sec", 0.1);
    holosoma_prepare_hold_sec_ = scalar_or<double>(mimic, "prepare_hold_sec", 0.0);
    holosoma_policy_warmup_sec_ = scalar_or<double>(mimic, "policy_warmup_sec", 0.05);
    holosoma_max_target_jump_ = scalar_or<double>(mimic, "max_target_jump", 1.2);
    holosoma_stop_on_large_jump_ = scalar_or<bool>(mimic, "tracking_protection", true);
    holosoma_wall_time_timestep_ = scalar_or<bool>(mimic, "wall_time_timestep", true);
  }

void GenericPolicyPlayer::post_load_beyondmimic() {
    obs_input_name_ = obs_input_name_.empty() ? "obs" : obs_input_name_;
    if (time_input_name_.empty()) time_input_name_ = "time_step";
    onnx_.require_inputs({obs_input_name_, time_input_name_});
    onnx_.require_outputs({"actions", "joint_pos", "joint_vel", "body_quat_w"});
    reset_beyond_controller();
  }

void GenericPolicyPlayer::post_load_holosoma() {
    obs_input_name_ = obs_input_name_.empty() ? "obs" : obs_input_name_;
    if (time_input_name_.empty()) time_input_name_ = "time_step";
    onnx_.require_inputs({obs_input_name_, time_input_name_});
    onnx_.require_outputs({"actions", "joint_pos", "joint_vel", "ref_quat_xyzw"});
    reset_holosoma_motion_buffers();
  }

void GenericPolicyPlayer::load_beyond_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("BeyondMimic params not found: " + params_path_.string());
    const YAML::Node y = YAML::LoadFile(params_path_.string());
    num_actions_ = yamlc::required_scalar<int>(y, "policy.num_actions");
    num_obs_ = yamlc::required_scalar<int>(y, "policy.num_obs");
    if (num_actions_ != motor_count_) {
      throw std::runtime_error("BeyondMimic policy.num_actions must match spec robot.motor_count");
    }
    if (configured_obs_dim_ > 0 && num_obs_ != configured_obs_dim_) {
      throw std::runtime_error("BeyondMimic policy.num_obs does not match spec observation.obs_dim");
    }
    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.policy_joint_order", motor_count_);
    beyond_waist_joint_idx_ = {
        required_joint_index(joint_order_, "waist_yaw_joint"),
        required_joint_index(joint_order_, "waist_roll_joint"),
        required_joint_index(joint_order_, "waist_pitch_joint"),
    };
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
    default_q_ = yamlc::required_vector_size<float>(y, "robot.default_angles", motor_count_);
    joint_index_ = yamlc::required_vector_size<int>(y, "robot.joint_map", motor_count_);
    yamlc::require_permutation("robot.joint_map", joint_index_, motor_count_);
    action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
    if (max_frames_ <= 0) throw std::runtime_error("BeyondMimic generic spec requires runtime.max_frames");
    beyond_action_.assign(num_actions_, 0.f);
    beyond_ref_joint_pos_.assign(num_actions_, 0.f);
    beyond_ref_joint_vel_.assign(num_actions_, 0.f);
    beyond_ref_body_quat_w_.assign(beyond_ref_body_count_ * 4, 0.f);
  }

void GenericPolicyPlayer::load_holosoma_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("Holosoma params not found: " + params_path_.string());
    const YAML::Node y = YAML::LoadFile(params_path_.string());
    num_actions_ = yamlc::required_scalar<int>(y, "policy.num_actions");
    num_obs_ = yamlc::required_scalar<int>(y, "policy.num_obs");
    if (num_actions_ != motor_count_) {
      throw std::runtime_error("Holosoma policy.num_actions must match spec robot.motor_count");
    }
    if (configured_obs_dim_ > 0 && num_obs_ != configured_obs_dim_) {
      throw std::runtime_error("Holosoma policy.num_obs does not match spec observation.obs_dim");
    }
    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.policy_joint_order", motor_count_);
    default_q_ = yamlc::required_vector_size<float>(y, "robot.default_angles", motor_count_);
    action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", motor_count_);
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
    joint_index_ = yamlc::required_vector_size<int>(y, "robot.joint_map", motor_count_);
    yamlc::require_permutation("robot.joint_map", joint_index_, motor_count_);
    holosoma_dof_names_ = yamlc::required_vector_size<std::string>(y, "robot.policy_joint_order", motor_count_);
    holosoma_stiff_startup_pos_ = yamlc::required_vector_size<float>(y, "action.stiff_startup_pos", motor_count_);
    holosoma_stiff_startup_kp_ = yamlc::required_vector_size<float>(y, "action.stiff_startup_kp", motor_count_);
    holosoma_stiff_startup_kd_ = yamlc::required_vector_size<float>(y, "action.stiff_startup_kd", motor_count_);
    validate_joint_index();
    if (holosoma_motion_end_step_ <= 0) {
      holosoma_motion_end_step_ = yamlc::required_scalar<int>(y, "motion.motion_end_step");
    }
    holosoma_last_policy_action_.assign(motor_count_, 0.f);
    holosoma_motion_joint_pos_.assign(motor_count_, 0.f);
    holosoma_motion_joint_vel_.assign(motor_count_, 0.f);
    holosoma_initial_motion_pos_.assign(motor_count_, 0.f);
  }

void GenericPolicyPlayer::schema_summary_beyondmimic() {
    schema_summary_common();
    std::cout << "  beyondmimic_contract ref_body_count=" << beyond_ref_body_count_
              << " anchor_body_index=" << beyond_ref_anchor_idx_
              << " waist_joint_indices=" << beyond_waist_joint_idx_[0] << ","
              << beyond_waist_joint_idx_[1] << "," << beyond_waist_joint_idx_[2]
              << " warmup_steps=" << beyond_warmup_steps_
              << " initial_reference_step=" << beyond_initial_reference_step_
              << " policy_step_source=" << (beyond_use_lowstate_tick_ ? "lowstate_tick" : "wall_clock")
              << " policy_tick_ms=" << beyond_policy_tick_ms_ << "\n";
  }

void GenericPolicyPlayer::schema_summary_holosoma() {
    schema_summary_common();
    std::cout << "  holosoma_contract motion_start=" << holosoma_motion_start_step_
              << " motion_end=" << holosoma_motion_end_step_ << "\n";
  }

void GenericPolicyPlayer::check_only_beyondmimic() {
    std::cout << "  beyondmimic obs_dim=" << num_obs_ << " actions=" << beyond_action_.size()
              << " max_frames=" << max_frames_
              << " warmup_steps=" << beyond_warmup_steps_
              << " warmup_ms=" << (1000 * beyond_warmup_steps_ / std::max(1, inference_hz_))
              << " initial_reference_step=" << beyond_initial_reference_step_ << "\n";
  }

void GenericPolicyPlayer::check_only_holosoma() {
    std::cout << "  holosoma obs_dim=" << num_obs_ << " motion_start=" << holosoma_motion_start_step_
              << " motion_end=" << holosoma_motion_end_step_ << " prepare_target=" << holosoma_prepare_target_ << "\n";
  }

GenericPolicyPlayer::BeyondResult GenericPolicyPlayer::beyond_inference(const std::vector<float>& obs, float t) {
    check_size("beyond obs", static_cast<size_t>(num_obs_), obs.size());
    std::unordered_map<std::string, int> dims{{"actions", num_actions_}, {"joint_pos", num_actions_},
                                              {"joint_vel", num_actions_}, {"body_quat_w", -1}};
    auto out = onnx_.run_obs_time(obs, t, obs_input_name_, time_input_name_, dims);
    if (out["body_quat_w"].size() < static_cast<size_t>((beyond_ref_anchor_idx_ + 1) * 4)) {
      throw std::runtime_error("BeyondMimic body_quat_w too small for anchor index " +
                               std::to_string(beyond_ref_anchor_idx_) + ": got " +
                               std::to_string(out["body_quat_w"].size()) + " floats");
    }
    return {out["actions"], out["joint_pos"], out["joint_vel"], out["body_quat_w"]};
  }

void GenericPolicyPlayer::reset_beyond_controller() {
    policy_counter_ = 0;
    beyond_motion_step_ = beyond_initial_reference_step_;
    beyond_init_to_world_ = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    auto r = beyond_inference(std::vector<float>(num_obs_, 0.f), static_cast<float>(beyond_motion_step_));
    beyond_action_.assign(num_actions_, 0.f);
    beyond_ref_joint_pos_ = r.joint_pos;
    beyond_ref_joint_vel_ = r.joint_vel;
    beyond_ref_body_quat_w_ = r.body_quat_w;
  }

bool GenericPolicyPlayer::build_beyond_target(const GenericPolicyPlayer::RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd) {
    CommandTimingScope timing_scope(s.stamp, s.seq);
    if (beyond_motion_step_ >= static_cast<int64_t>(max_frames_)) return false;
    if (s.q.size() < static_cast<size_t>(num_actions_)) throw std::runtime_error("lowstate motor count too small");
    std::vector<float> qj(num_actions_, 0.f), dqj(num_actions_, 0.f);
    std::vector<float> q_abs(num_actions_, 0.f);
    for (int i = 0; i < num_actions_; ++i) {
      const int mj = joint_index_[i];
      q_abs[i] = s.q.at(mj);
      qj[i] = s.q.at(mj) - default_q_[i];
      dqj[i] = s.dq.at(mj);
    }
    std::array<float, 4> robot_q{s.quat[0], s.quat[1], s.quat[2], s.quat[3]};
    robot_q = quat_mul_legacy(
        robot_q,
        quat_mul_legacy(
            axis_quat(q_abs.at(beyond_waist_joint_idx_[0]), 'z'),
            quat_mul_legacy(axis_quat(q_abs.at(beyond_waist_joint_idx_[1]), 'x'),
                            axis_quat(q_abs.at(beyond_waist_joint_idx_[2]), 'y'))));
    std::array<float, 4> ref{beyond_ref_body_quat_w_[beyond_ref_anchor_idx_ * 4],
                             beyond_ref_body_quat_w_[beyond_ref_anchor_idx_ * 4 + 1],
                             beyond_ref_body_quat_w_[beyond_ref_anchor_idx_ * 4 + 2],
                             beyond_ref_body_quat_w_[beyond_ref_anchor_idx_ * 4 + 3]};
    if (policy_counter_ == 0 && beyond_align_initial_yaw_) {
      beyond_init_to_world_ = mat_mul(matrix_from_quat(yaw_quat(robot_q)), mat_transpose(matrix_from_quat(yaw_quat(ref))));
    }
    if (policy_counter_ < beyond_warmup_steps_) {
      hold_current_pose(s, q, kp, kd);
      ++policy_counter_;
      return true;
    }
    const auto anchor_b = mat_mul(mat_mul(mat_transpose(matrix_from_quat(robot_q)), beyond_init_to_world_), matrix_from_quat(ref));
    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(num_obs_));
    append(obs, beyond_ref_joint_pos_);
    append(obs, beyond_ref_joint_vel_);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 2; ++c) obs.push_back(anchor_b[r * 3 + c]);
    append(obs, s.gyro);
    append(obs, qj);
    append(obs, dqj);
    append(obs, beyond_action_);
    check_size("beyond obs", static_cast<size_t>(num_obs_), obs.size());
    // MJLab evaluates the actor against the current reference frame, then
    // advances the command time after the environment step.  Keep the ONNX
    // time input aligned with that same 0,1,2,... sequence.
    const int64_t current_motion_step = std::min<int64_t>(beyond_motion_step_, max_frames_ - 1);
    auto res = beyond_inference(obs, static_cast<float>(current_motion_step));
    beyond_action_ = res.actions;
    beyond_ref_joint_pos_ = res.joint_pos;
    beyond_ref_joint_vel_ = res.joint_vel;
    beyond_ref_body_quat_w_ = res.body_quat_w;
    q.assign(motor_count_, 0.f);
    kp.assign(motor_count_, 0.f);
    kd.assign(motor_count_, 0.f);
    for (int i = 0; i < num_actions_; ++i) {
      const int mj = joint_index_[i];
      q[mj] = beyond_action_[i] * action_scale_[i] + default_q_[i];
      kp[mj] = kp_[i];
      kd[mj] = kd_[i];
    }
    ++policy_counter_;
    ++beyond_motion_step_;
    if (beyond_motion_step_ < static_cast<int64_t>(max_frames_)) {
      // The reference command advances with the MJLab environment step.  The
      // actor output above belongs to frame N, while the next observation must
      // carry frame N+1 even though the embedded reference is only exposed as
      // an ONNX output.  Refresh that output without reusing its action.
      const auto next_ref = beyond_inference(std::vector<float>(num_obs_, 0.f),
                                             static_cast<float>(beyond_motion_step_));
      beyond_ref_joint_pos_ = next_ref.joint_pos;
      beyond_ref_joint_vel_ = next_ref.joint_vel;
      beyond_ref_body_quat_w_ = next_ref.body_quat_w;
    }
    return true;
  }

void GenericPolicyPlayer::control_loop_beyondmimic() {
    reset_beyond_controller();
    const int publish_hz = std::max(inference_hz_, publish_hz_);
    const auto policy_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, inference_hz_)));
    const auto publish_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / publish_hz));
    auto s = snapshot();
    std::vector<float> q, kp, kd;
    hold_current_pose(s, q, kp, kd);
    publish(q, kp, kd, s.mode_machine, false, CommandPhase::kHold);
    if (!ready_file_.empty()) {
      touch_file(ready_file_);
      std::cout << player_log_prefix() << " ready file touched: " << ready_file_ << "\n";
    }

    bool preview_committed = false;
    if (preview_mode_enabled()) {
      const auto saved_policy_counter = policy_counter_;
      const auto saved_motion_step = beyond_motion_step_;
      const auto saved_init_to_world = beyond_init_to_world_;
      const auto saved_action = beyond_action_;
      const auto saved_ref_joint_pos = beyond_ref_joint_pos_;
      const auto saved_ref_joint_vel = beyond_ref_joint_vel_;
      const auto saved_ref_body_quat = beyond_ref_body_quat_w_;
      auto restore_preview_state = [&]() {
        policy_counter_ = saved_policy_counter;
        beyond_motion_step_ = saved_motion_step;
        beyond_init_to_world_ = saved_init_to_world;
        beyond_action_ = saved_action;
        beyond_ref_joint_pos_ = saved_ref_joint_pos;
        beyond_ref_joint_vel_ = saved_ref_joint_vel;
        beyond_ref_body_quat_w_ = saved_ref_body_quat;
      };

      for (int step = 0; step <= beyond_warmup_steps_; ++step) {
        if (!build_beyond_target(s, q, kp, kd)) {
          restore_preview_state();
          throw std::runtime_error("BeyondMimic ended before producing preview candidate");
        }
      }
      const auto decision = wait_preview_decision(
          q, kp, kd, s.mode_machine, publish_hz, beyond_warmup_steps_);
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
    } else {
      const auto prestart_state = s;
      // The official controller may continue making small posture corrections
      // while the payload is hot-standby.  Keep the standby command aligned
      // with the newest LowState so SwitchToUserCtrl never exposes an old pose
      // target, even briefly, before the warmup hold takes over.
      wait_start_file(q, kp, kd, publish_hz, {}, true);
      if (g_stop_requested.load()) return;
      const auto handoff_state = snapshot();
      double max_q_drift = 0.0;
      int max_q_drift_motor = -1;
      const size_t q_count = std::min(prestart_state.q.size(), handoff_state.q.size());
      for (size_t index = 0; index < q_count; ++index) {
        const double drift = std::abs(static_cast<double>(handoff_state.q[index]) -
                                      prestart_state.q[index]);
        if (drift > max_q_drift) {
          max_q_drift = drift;
          max_q_drift_motor = static_cast<int>(index);
        }
      }
      double dq_abs_max = 0.0;
      for (float value : handoff_state.dq) {
        dq_abs_max = std::max(dq_abs_max, std::abs(static_cast<double>(value)));
      }
      double gyro_abs_max = 0.0;
      for (float value : handoff_state.gyro) {
        gyro_abs_max = std::max(gyro_abs_max, std::abs(static_cast<double>(value)));
      }
      std::cout << player_log_prefix() << " handoff state"
                << " max_q_drift_from_ready=" << max_q_drift
                << " max_q_drift_motor=" << max_q_drift_motor
                << " dq_abs_max=" << dq_abs_max
                << " gyro_abs_max=" << gyro_abs_max
                << " source_tick=" << handoff_state.source_tick_ms << "\n";
      event_log_.log("handoff_state",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_number("max_q_drift_from_ready", max_q_drift),
                      event_int("max_q_drift_motor", max_q_drift_motor),
                      event_number("dq_abs_max", dq_abs_max),
                      event_number("gyro_abs_max", gyro_abs_max),
                      event_int("source_tick_ms", handoff_state.source_tick_ms)});
    }
    auto start = std::chrono::steady_clock::now();
    auto next_policy = preview_committed ? start + policy_dt : start;
    auto next_publish = start;
    int64_t inference_count = 0;
    int64_t publish_count = 0;
    int64_t policy_deadline_miss = 0;
    int64_t publish_deadline_miss = 0;
    bool finished = false;
    bool startup_interp_active = false;
    bool startup_interp_started = false;
    bool first_active_target_logged = false;
    double inference_ms_total = 0.0;
    double inference_ms_max = 0.0;
    std::vector<float> startup_interp_from, startup_interp_to;
    std::vector<float> startup_interp_from_kp, startup_interp_to_kp;
    std::vector<float> startup_interp_from_kd, startup_interp_to_kd;
    auto startup_interp_start = start;
    uint32_t last_policy_tick_ms = snapshot().source_tick_ms;
    auto model_joint_for_motor = [&](int motor_index) {
      for (int model_index = 0; model_index < static_cast<int>(joint_index_.size()); ++model_index) {
        if (joint_index_[model_index] == motor_index) return model_index;
      }
      return -1;
    };
    auto csv_values = [](const auto& values) {
      std::ostringstream stream;
      stream << std::setprecision(9);
      for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ',';
        stream << values[index];
      }
      return stream.str();
    };
    std::cout << player_log_prefix() << " start embedded_mimic_time/beyondmimic model=" << runtime_model_path_.string()
              << " max_frames=" << max_frames_ << " hz=" << inference_hz_ << " publish_hz=" << publish_hz << "\n";
    while (!g_stop_requested.load()) {
      const auto now = std::chrono::steady_clock::now();
      if (duration_sec_ > 0.0 &&
          std::chrono::duration<double>(now - start).count() >= duration_sec_) {
        finished = true;
        break;
      }
      auto policy_state = snapshot();
      const uint32_t tick_elapsed = policy_state.source_tick_ms - last_policy_tick_ms;
      const bool tick_policy_due = inference_count == 0 || tick_elapsed >= beyond_policy_tick_ms_;
      const bool policy_due = beyond_use_lowstate_tick_ ? tick_policy_due : now >= next_policy;
      if (policy_due) {
        if (beyond_use_lowstate_tick_) {
          if (inference_count > 0 && tick_elapsed > beyond_policy_tick_ms_ * 2) {
            policy_deadline_miss +=
                static_cast<int64_t>(tick_elapsed / beyond_policy_tick_ms_) - 1;
          }
          last_policy_tick_ms = policy_state.source_tick_ms;
        } else {
          policy_deadline_miss += advance_deadline(next_policy, policy_dt, now);
        }
        const double startup_elapsed = std::chrono::duration<double>(now - startup_interp_start).count();
        bool startup_interp_just_finished = false;
        if (startup_interp_active && startup_elapsed < startup_interp_sec_) {
          // Keep the first policy target fixed while the robot ramps from its
          // reset/default pose.  Advancing the embedded motion here would
          // skip the opening frame before the ramp has completed.
        } else {
          if (startup_interp_active) {
            q = startup_interp_to;
            kp = startup_interp_to_kp;
            kd = startup_interp_to_kd;
            startup_interp_active = false;
            startup_interp_just_finished = true;
            double max_target_error = 0.0;
            int max_target_error_motor = -1;
            for (size_t index = 0; index < q.size() && index < policy_state.q.size(); ++index) {
              const double error = std::abs(static_cast<double>(q[index]) - policy_state.q[index]);
              if (error > max_target_error) {
                max_target_error = error;
                max_target_error_motor = static_cast<int>(index);
              }
            }
            double dq_abs_max = 0.0;
            for (float value : policy_state.dq) {
              dq_abs_max = std::max(dq_abs_max, std::abs(static_cast<double>(value)));
            }
            double gyro_abs_max = 0.0;
            for (float value : policy_state.gyro) {
              gyro_abs_max = std::max(gyro_abs_max, std::abs(static_cast<double>(value)));
            }
            std::cout << player_log_prefix() << " startup interpolation completed"
                      << " duration_sec=" << startup_interp_sec_
                      << " max_target_error=" << max_target_error
                      << " max_target_error_motor=" << max_target_error_motor
                      << " dq_abs_max=" << dq_abs_max
                      << " gyro_abs_max=" << gyro_abs_max
                      << " source_tick=" << policy_state.source_tick_ms << "\n";
            event_log_.log("startup_interpolation_completed",
                           {event_string("player", player_label()),
                            event_string("template", template_),
                            event_string("variant", variant_),
                            event_number("duration_sec", startup_interp_sec_),
                            event_number("max_target_error", max_target_error),
                            event_int("max_target_error_motor", max_target_error_motor),
                            event_number("dq_abs_max", dq_abs_max),
                            event_number("gyro_abs_max", gyro_abs_max),
                            event_int("source_tick_ms", policy_state.source_tick_ms)});
          }
          s = policy_state;
          std::vector<float> next_q, next_kp, next_kd;
          const auto t0 = std::chrono::steady_clock::now();
          const bool active_candidate = policy_counter_ >= beyond_warmup_steps_;
          const int64_t active_frame = beyond_motion_step_;
          // Commit the interpolation endpoint for this cycle.  Build the
          // next policy target on the following cycle so frame 0 is not
          // replaced by frame 1 at the exact ramp boundary.
          if (!startup_interp_just_finished && !build_beyond_target(s, next_q, next_kp, next_kd)) {
            finished = true;
          } else if (!startup_interp_just_finished && !next_q.empty()) {
            const int64_t active_step = policy_counter_ - beyond_warmup_steps_;
            if (active_candidate && !first_active_target_logged) {
              double max_abs_error = 0.0;
              int max_error_index = -1;
              for (size_t index = 0; index < next_q.size() && index < s.q.size(); ++index) {
                const double error = std::abs(static_cast<double>(next_q[index]) - s.q[index]);
                if (error > max_abs_error) {
                  max_abs_error = error;
                  max_error_index = static_cast<int>(index);
                }
              }
              const auto rpy = quat_rpy(s.quat);
              constexpr double kRadToDeg = 57.29577951308232;
              const double target_min = *std::min_element(next_q.begin(), next_q.end());
              const double target_max = *std::max_element(next_q.begin(), next_q.end());
              const int max_error_model = model_joint_for_motor(max_error_index);
              const std::string max_error_joint =
                  max_error_model >= 0 && max_error_model < static_cast<int>(joint_order_.size())
                      ? joint_order_[max_error_model] : "unknown";
              const double current_q = max_error_index >= 0 && max_error_index < static_cast<int>(s.q.size())
                  ? s.q[max_error_index] : 0.0;
              const double target_q = max_error_index >= 0 && max_error_index < static_cast<int>(next_q.size())
                  ? next_q[max_error_index] : 0.0;
              std::vector<std::string> motor_joint_names(static_cast<size_t>(motor_count_), "unknown");
              for (int model_index = 0; model_index < static_cast<int>(joint_index_.size()); ++model_index) {
                const int motor_index = joint_index_[model_index];
                if (motor_index >= 0 && motor_index < motor_count_ &&
                    model_index < static_cast<int>(joint_order_.size())) {
                  motor_joint_names[motor_index] = joint_order_[model_index];
                }
              }
              const std::string joint_names_csv = csv_values(motor_joint_names);
              const std::string current_q_csv = csv_values(s.q);
              const std::string target_q_csv = csv_values(next_q);
              const std::string current_dq_csv = csv_values(s.dq);
              const std::string gyro_csv = csv_values(s.gyro);
              const std::string quat_csv = csv_values(s.quat);
              std::cout << player_log_prefix() << " first active target"
                        << " frame=" << active_frame
                        << " max_q_error=" << max_abs_error
                        << " max_q_error_motor=" << max_error_index
                        << " target_min=" << target_min
                        << " target_max=" << target_max
                        << " max_q_error_joint=" << max_error_joint
                        << " current_q=" << current_q
                        << " target_q=" << target_q
                        << " imu_rpy_deg=" << (rpy[0] * kRadToDeg) << ","
                        << (rpy[1] * kRadToDeg) << ","
                        << (rpy[2] * kRadToDeg)
                        << " source_tick=" << policy_state.source_tick_ms << "\n";
              std::cout << player_log_prefix() << " first active state"
                        << " joint_names=" << joint_names_csv
                        << " current_q=" << current_q_csv
                        << " target_q=" << target_q_csv
                        << " current_dq=" << current_dq_csv
                        << " gyro=" << gyro_csv
                        << " quat_wxyz=" << quat_csv
                        << " source_tick=" << policy_state.source_tick_ms << "\n";
              event_log_.log("first_active_target",
                             {event_string("player", player_label()),
                              event_string("template", template_),
                              event_string("variant", variant_),
                              event_int("motion_frame", active_frame),
                              event_number("max_q_error", max_abs_error),
                              event_int("max_q_error_motor", max_error_index),
                              event_number("target_min", target_min),
                              event_number("target_max", target_max),
                              event_string("max_q_error_joint", max_error_joint),
                              event_number("current_q", current_q),
                              event_number("target_q", target_q),
                              event_number("imu_roll_deg", rpy[0] * kRadToDeg),
                              event_number("imu_pitch_deg", rpy[1] * kRadToDeg),
                              event_number("imu_yaw_deg", rpy[2] * kRadToDeg),
                              event_int("source_tick_ms", policy_state.source_tick_ms)});
              event_log_.log("first_active_state",
                             {event_string("player", player_label()),
                              event_string("template", template_),
                              event_string("variant", variant_),
                              event_string("joint_names_csv", joint_names_csv),
                              event_string("current_q_csv", current_q_csv),
                              event_string("target_q_csv", target_q_csv),
                              event_string("current_dq_csv", current_dq_csv),
                              event_string("gyro_csv", gyro_csv),
                              event_string("quat_wxyz_csv", quat_csv),
                              event_int("source_tick_ms", policy_state.source_tick_ms)});
              first_active_target_logged = true;
            }
            if (active_candidate &&
                (active_step == 4 || active_step == 9 || active_step == 24)) {
              double max_abs_error = 0.0;
              int max_error_index = -1;
              for (size_t index = 0; index < next_q.size() && index < s.q.size(); ++index) {
                const double error = std::abs(static_cast<double>(next_q[index]) - s.q[index]);
                if (error > max_abs_error) {
                  max_abs_error = error;
                  max_error_index = static_cast<int>(index);
                }
              }
              double action_abs_max = 0.0;
              for (float value : beyond_action_) {
                action_abs_max = std::max(action_abs_max, std::abs(static_cast<double>(value)));
              }
              double dq_abs_max = 0.0;
              for (float value : s.dq) {
                dq_abs_max = std::max(dq_abs_max, std::abs(static_cast<double>(value)));
              }
              double gyro_abs_max = 0.0;
              for (float value : s.gyro) {
                gyro_abs_max = std::max(gyro_abs_max, std::abs(static_cast<double>(value)));
              }
              const auto rpy = quat_rpy(s.quat);
              constexpr double kRadToDeg = 57.29577951308232;
              const int max_error_model = model_joint_for_motor(max_error_index);
              const std::string max_error_joint =
                  max_error_model >= 0 && max_error_model < static_cast<int>(joint_order_.size())
                      ? joint_order_[max_error_model] : "unknown";
              const double current_q = max_error_index >= 0 && max_error_index < static_cast<int>(s.q.size())
                  ? s.q[max_error_index] : 0.0;
              const double target_q = max_error_index >= 0 && max_error_index < static_cast<int>(next_q.size())
                  ? next_q[max_error_index] : 0.0;
              std::cout << player_log_prefix() << " early diagnostics"
                        << " frame=" << active_frame
                        << " max_q_error=" << max_abs_error
                        << " max_q_error_motor=" << max_error_index
                        << " max_q_error_joint=" << max_error_joint
                        << " current_q=" << current_q
                        << " target_q=" << target_q
                        << " action_abs_max=" << action_abs_max
                        << " dq_abs_max=" << dq_abs_max
                        << " gyro_abs_max=" << gyro_abs_max
                        << " imu_rpy_deg=" << (rpy[0] * kRadToDeg) << ","
                        << (rpy[1] * kRadToDeg) << ","
                        << (rpy[2] * kRadToDeg)
                        << " source_tick=" << policy_state.source_tick_ms << "\n";
              event_log_.log("early_runtime_diagnostics",
                             {event_string("player", player_label()),
                              event_string("template", template_),
                              event_string("variant", variant_),
                              event_int("motion_frame", active_frame),
                              event_number("max_q_error", max_abs_error),
                              event_int("max_q_error_motor", max_error_index),
                              event_string("max_q_error_joint", max_error_joint),
                              event_number("current_q", current_q),
                              event_number("target_q", target_q),
                              event_number("action_abs_max", action_abs_max),
                              event_number("dq_abs_max", dq_abs_max),
                              event_number("gyro_abs_max", gyro_abs_max),
                              event_number("imu_roll_deg", rpy[0] * kRadToDeg),
                              event_number("imu_pitch_deg", rpy[1] * kRadToDeg),
                              event_number("imu_yaw_deg", rpy[2] * kRadToDeg),
                              event_int("source_tick_ms", policy_state.source_tick_ms)});
            }
            if (!startup_interp_started && active_candidate && startup_interp_sec_ > 1e-6) {
              startup_interp_from = q;
              startup_interp_to = next_q;
              startup_interp_from_kp = kp;
              startup_interp_from_kd = kd;
              startup_interp_to_kp = next_kp;
              startup_interp_to_kd = next_kd;
              startup_interp_start = now;
              startup_interp_active = true;
              startup_interp_started = true;
              std::cout << player_log_prefix() << " startup interpolation"
                        << " frame=" << active_frame
                        << " duration_sec=" << startup_interp_sec_ << "\n";
              event_log_.log("startup_interpolation_started",
                             {event_string("player", player_label()),
                              event_string("template", template_),
                              event_string("variant", variant_),
                              event_int("motion_frame", active_frame),
                              event_number("duration_sec", startup_interp_sec_)});
              kp = startup_interp_from_kp;
              kd = startup_interp_from_kd;
            } else {
              q = std::move(next_q);
              kp = std::move(next_kp);
              kd = std::move(next_kd);
            }
          }
          const auto t1 = std::chrono::steady_clock::now();
          const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
          ++inference_count;
          inference_ms_total += ms;
          inference_ms_max = std::max(inference_ms_max, ms);
          if (stats_interval_ > 0 && inference_count % stats_interval_ == 0) {
            const auto rpy = quat_rpy(policy_state.quat);
            constexpr double kRadToDeg = 57.29577951308232;
            std::cout << player_log_prefix() << " metrics frame=" << beyond_motion_step_
                      << " infer_ms=" << ms
                      << " policy_deadline_miss=" << policy_deadline_miss
                      << " publish_deadline_miss=" << publish_deadline_miss
                      << " imu_rpy_deg=" << (rpy[0] * kRadToDeg) << ","
                      << (rpy[1] * kRadToDeg) << ","
                      << (rpy[2] * kRadToDeg)
                      << " source_tick=" << policy_state.source_tick_ms << "\n";
            event_log_.log("runtime_metrics",
                           {event_string("player", player_label()),
                            event_string("template", template_),
                            event_string("variant", variant_),
                            event_int("step", inference_count),
                            event_number("infer_ms_last", ms),
                            event_int("policy_deadline_miss", policy_deadline_miss),
                            event_int("publish_deadline_miss", publish_deadline_miss)});
          }
        }
      }
      if (now >= next_publish) {
        publish_deadline_miss += advance_deadline(next_publish, publish_dt, now);
        s = snapshot();
        std::vector<float> publish_q = q;
        if (startup_interp_active && startup_interp_from.size() == startup_interp_to.size()) {
          const double alpha = std::chrono::duration<double>(now - startup_interp_start).count() /
                               std::max(startup_interp_sec_, 1e-6);
          publish_q = lerp_vec(startup_interp_from, startup_interp_to, alpha);
          kp = lerp_vec(startup_interp_from_kp, startup_interp_to_kp, alpha);
          kd = lerp_vec(startup_interp_from_kd, startup_interp_to_kd, alpha);
        }
        publish(publish_q, kp, kd, s.mode_machine);
        ++publish_count;
      }
      if (finished) break;
      const auto sleep_until = beyond_use_lowstate_tick_ ? next_publish : std::min(next_policy, next_publish);
      if (sleep_until > std::chrono::steady_clock::now()) std::this_thread::sleep_until(sleep_until);
    }
    if (finished && !g_stop_requested.load() && !handoff_complete_file_.empty()) {
      if (last_published_q_.size() == static_cast<size_t>(motor_count_)) {
        q = last_published_q_;
        kp = last_published_kp_;
        kd = last_published_kd_;
      }
      event_log_.log("handoff_hold_started",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_string("complete_file", handoff_complete_file_),
                      event_string("release_file", handoff_release_file_)});
      s = snapshot();
      publish(q, kp, kd, s.mode_machine, false, CommandPhase::kHold);
      ++publish_count;
      touch_file(handoff_complete_file_);
      auto next_hold_publish = std::chrono::steady_clock::now() + publish_dt;
      while (!g_stop_requested.load() && !fs::exists(handoff_release_file_)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_hold_publish) {
          s = snapshot();
          publish(q, kp, kd, s.mode_machine, false, CommandPhase::kHold);
          ++publish_count;
          next_hold_publish += publish_dt;
          if (next_hold_publish < now) next_hold_publish = now + publish_dt;
        }
        if (next_hold_publish > std::chrono::steady_clock::now()) {
          std::this_thread::sleep_until(next_hold_publish);
        }
      }
      event_log_.log("handoff_hold_stopped",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_bool("released", fs::exists(handoff_release_file_)),
                      event_string("reason", g_stop_requested.load() ? "signal" : "release_file")});
    }
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    std::cout << player_log_prefix() << " beyondmimic stats inference=" << inference_count
              << " publish=" << publish_count
              << " inference_ms_avg=" << (inference_count > 0 ? inference_ms_total / inference_count : 0.0)
              << " inference_ms_max=" << inference_ms_max
              << " policy_deadline_miss=" << policy_deadline_miss
              << " publish_deadline_miss=" << publish_deadline_miss << "\n";
    event_log_.log("runtime_metrics",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_int("step", inference_count),
                    event_int("publish_count", publish_count),
                    event_int("policy_deadline_miss", policy_deadline_miss),
                    event_int("publish_deadline_miss", publish_deadline_miss),
                    event_string("phase", "final")});
    std::cout << player_log_prefix() << " completed beyondmimic\n";
  }

GenericPolicyPlayer::HoloResult GenericPolicyPlayer::holosoma_inference(const std::vector<float>& obs, float timestep) {
    check_size("holosoma obs", static_cast<size_t>(num_obs_), obs.size());
    std::unordered_map<std::string, int> dims{{"actions", motor_count_}, {"joint_pos", motor_count_},
                                              {"joint_vel", motor_count_}, {"ref_quat_xyzw", 4}};
    auto out = onnx_.run_obs_time(obs, timestep, obs_input_name_, time_input_name_, dims);
    const auto& q = out["ref_quat_xyzw"];
    return {out["actions"], out["joint_pos"], out["joint_vel"], {q[0], q[1], q[2], q[3]}};
  }

std::vector<float> GenericPolicyPlayer::model_to_motor(const std::vector<float>& model_vec, float fill ) const {
    check_size("model_to_motor", motor_count_, model_vec.size());
    std::vector<float> motor(motor_count_, fill);
    for (int model_i = 0; model_i < motor_count_; ++model_i) motor[joint_index_[model_i]] = model_vec[model_i];
    return motor;
  }

std::vector<float> GenericPolicyPlayer::motor_to_model(const std::vector<float>& motor_vec) const {
    check_size("motor_to_model", motor_count_, motor_vec.size());
    std::vector<float> model(motor_count_, 0.f);
    for (int model_i = 0; model_i < motor_count_; ++model_i) model[model_i] = motor_vec[joint_index_[model_i]];
    return model;
  }

std::vector<float> GenericPolicyPlayer::clamp_model_q(const std::vector<float>& q_model) const {
    check_size("clamp_model_q", motor_count_, q_model.size());
    std::vector<float> out(motor_count_, 0.f);
    for (int i = 0; i < motor_count_; ++i) out[i] = std::clamp(q_model[i], lower_limits_[i], upper_limits_[i]);
    return out;
  }

void GenericPolicyPlayer::reset_holosoma_motion_buffers() {
    HoloResult r = holosoma_inference(std::vector<float>(num_obs_, 0.f), static_cast<float>(holosoma_motion_start_step_));
    holosoma_initial_motion_pos_ = clamp_model_q(r.joint_pos);
    holosoma_motion_joint_pos_ = std::move(r.joint_pos);
    holosoma_motion_joint_vel_ = std::move(r.joint_vel);
    holosoma_ref_quat_xyzw_ = r.ref_quat_xyzw;
    holosoma_last_policy_action_.assign(motor_count_, 0.f);
    holosoma_current_timestep_ = holosoma_motion_start_step_;
  }

std::array<float, 4> GenericPolicyPlayer::torso_quat_from_model_q(const std::array<float, 4>& base_quat, const std::vector<float>& q_model) const {
    if (q_model.size() < motor_count_) return normalize_quat(base_quat);
    return quat_mul(quat_mul(quat_mul(normalize_quat(base_quat), axis_quat(q_model[12], 'z')),
                             axis_quat(q_model[13], 'x')),
                    axis_quat(q_model[14], 'y'));
  }

std::array<float, 4> GenericPolicyPlayer::torso_quat_from_state(const GenericPolicyPlayer::RobotState& s) const {
    if (s.q.size() < motor_count_) return normalize_quat(s.quat);
    return torso_quat_from_model_q(s.quat, motor_to_model(s.q));
  }

std::vector<float> GenericPolicyPlayer::build_holosoma_obs(const GenericPolicyPlayer::RobotState& s) const {
    if (s.q.size() < motor_count_ || s.dq.size() < motor_count_) throw std::runtime_error("lowstate motor count too small");
    const std::vector<float> q_model = motor_to_model(s.q);
    const std::vector<float> dq_model = motor_to_model(s.dq);
    const auto robot_ref_ori = remove_yaw_offset(torso_quat_from_model_q(s.quat, q_model), holosoma_robot_yaw_offset_);
    const auto motion_ref_ori = remove_yaw_offset(xyzw_to_wxyz(holosoma_ref_quat_xyzw_), holosoma_motion_yaw_offset_);
    const auto rel = quat_mul(quat_inverse(robot_ref_ori), motion_ref_ori);
    const auto R = matrix_from_quat(rel);
    std::vector<float> dof_pos(motor_count_, 0.f), dof_vel(motor_count_, 0.f);
    for (int i = 0; i < motor_count_; ++i) {
      dof_pos[i] = q_model[i] - default_q_[i];
      dof_vel[i] = dq_model[i];
    }
    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(num_obs_));
    append(obs, holosoma_last_policy_action_);
    append(obs, s.gyro);
    append(obs, dof_pos);
    append(obs, dof_vel);
    append(obs, holosoma_motion_joint_pos_);
    append(obs, holosoma_motion_joint_vel_);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 2; ++c) obs.push_back(R[r * 3 + c]);
    check_size("holosoma obs", static_cast<size_t>(num_obs_), obs.size());
    return obs;
  }

int GenericPolicyPlayer::holosoma_timestep_from_wall_clock() const {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - holosoma_motion_clip_start_time_).count();
    const int elapsed_steps = static_cast<int>(std::floor(std::max(0.0, elapsed) * static_cast<double>(std::max(1, inference_hz_))));
    return holosoma_motion_start_step_ + elapsed_steps;
  }

GenericPolicyPlayer::JumpStats GenericPolicyPlayer::target_jump_stats(const std::vector<float>& a, const std::vector<float>& b) const {
    JumpStats out;
    if (a.size() < motor_count_ || b.size() < motor_count_) return out;
    for (int model_i = 0; model_i < motor_count_; ++model_i) {
      const int motor_i = joint_index_[model_i];
      const double d = std::abs(static_cast<double>(b[motor_i]) - static_cast<double>(a[motor_i]));
      if (d > out.all) { out.all = d; out.all_model = model_i; out.all_motor = motor_i; }
      if (model_i < 15 && d > out.lower_body) { out.lower_body = d; out.lower_body_model = model_i; out.lower_body_motor = motor_i; }
    }
    return out;
  }

std::string GenericPolicyPlayer::joint_desc(int model_i, int motor_i) const {
    std::string name = (model_i >= 0 && model_i < static_cast<int>(holosoma_dof_names_.size())) ? holosoma_dof_names_[model_i] : "unknown";
    return name + "(model=" + std::to_string(model_i) + ",motor=" + std::to_string(motor_i) + ")";
  }

bool GenericPolicyPlayer::check_holosoma_target_safety(const GenericPolicyPlayer::RobotState& state, const std::vector<float>& q_target, bool motion_clip_progressing) {
    if (!holosoma_stop_on_large_jump_) return true;
    const char* phase = motion_clip_progressing ? "motion_clip" : "policy_warmup";
    const JumpStats tracking = target_jump_stats(state.q, q_target);
    const auto rpy = quat_rpy(torso_quat_from_state(state));
    constexpr float kPi = 3.14159265358979323846f;
    if (tracking.all > holosoma_max_target_jump_ && holosoma_large_upper_body_tracking_warnings_ < 5) {
      ++holosoma_large_upper_body_tracking_warnings_;
      std::cerr << player_log_prefix() << " holosoma large q_target tracking error " << tracking.all
                << " rad at timestep " << holosoma_current_timestep_ << " phase=" << phase
                << " joint=" << joint_desc(tracking.all_model, tracking.all_motor)
                << "; lower_body=" << tracking.lower_body << " rad at "
                << joint_desc(tracking.lower_body_model, tracking.lower_body_motor)
                << "; torso_rpy_deg=[" << (rpy[0] * 180.0f / kPi) << "," << (rpy[1] * 180.0f / kPi)
                << "," << (rpy[2] * 180.0f / kPi) << "]; continuing.\n";
    }
    if (tracking.all > holosoma_max_target_jump_) return false;
    if (holosoma_have_prev_policy_target_) {
      const JumpStats step = target_jump_stats(holosoma_prev_policy_target_q_, q_target);
      if (step.all > holosoma_max_target_jump_ && holosoma_large_target_step_warnings_ < 5) {
        ++holosoma_large_target_step_warnings_;
        std::cerr << player_log_prefix() << " holosoma large consecutive q_target step " << step.all
                  << " rad at timestep " << holosoma_current_timestep_ << " phase=" << phase
                  << " joint=" << joint_desc(step.all_model, step.all_motor) << "; continuing.\n";
      }
      if (step.all > holosoma_max_target_jump_) return false;
    }
    return true;
  }

bool GenericPolicyPlayer::build_holosoma_policy_command(const GenericPolicyPlayer::RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                                     bool motion_clip_progressing) {
    CommandTimingScope timing_scope(s.stamp, s.seq);
    if (motion_clip_progressing && holosoma_wall_time_timestep_) {
      holosoma_current_timestep_ = holosoma_timestep_from_wall_clock();
    } else if (!motion_clip_progressing) {
      holosoma_current_timestep_ = holosoma_motion_start_step_;
    }
    if (motion_clip_progressing && holosoma_motion_end_step_ > 0 && holosoma_current_timestep_ >= holosoma_motion_end_step_) return false;

    auto r = holosoma_inference(build_holosoma_obs(s), static_cast<float>(holosoma_current_timestep_));
    holosoma_last_policy_action_.assign(motor_count_, 0.f);
    q.assign(motor_count_, 0.f);
    kp = model_to_motor(kp_);
    kd = model_to_motor(kd_);
    std::vector<float> model_q(motor_count_, 0.f);
    for (int i = 0; i < motor_count_; ++i) {
      const float action = std::clamp(r.actions[i], -100.f, 100.f);
      holosoma_last_policy_action_[i] = action;
      model_q[i] = default_q_[i] + action * action_scale_[i];
      model_q[i] = std::clamp(model_q[i], lower_limits_[i], upper_limits_[i]);
    }
    q = model_to_motor(model_q);
    if (!check_holosoma_target_safety(s, q, motion_clip_progressing)) return false;
    holosoma_prev_policy_target_q_ = q;
    holosoma_have_prev_policy_target_ = true;
    holosoma_motion_joint_pos_ = std::move(r.joint_pos);
    holosoma_motion_joint_vel_ = std::move(r.joint_vel);
    holosoma_ref_quat_xyzw_ = r.ref_quat_xyzw;
    if (!motion_clip_progressing) {
      holosoma_current_timestep_ = holosoma_motion_start_step_;
    } else if (holosoma_wall_time_timestep_) {
      holosoma_current_timestep_ = holosoma_timestep_from_wall_clock();
    } else {
      ++holosoma_current_timestep_;
    }
    return true;
  }

void GenericPolicyPlayer::holosoma_set_current_pose_hold() {
    const auto s = snapshot();
    holosoma_hold_q_ = s.q.size() >= motor_count_ ? s.q : model_to_motor(holosoma_stiff_startup_pos_);
    holosoma_hold_q_.resize(motor_count_, 0.f);
    holosoma_hold_kp_ = model_to_motor(holosoma_stiff_startup_kp_);
    holosoma_hold_kd_ = model_to_motor(holosoma_stiff_startup_kd_);
  }

void GenericPolicyPlayer::holosoma_prepare_startup_pose(int publish_hz) {
    if (holosoma_prepare_target_ == "none") {
      holosoma_set_current_pose_hold();
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / publish_hz));
    auto next = start;
    const double ramp_sec = std::max(0.0, holosoma_prepare_ramp_sec_);
    const double hold_sec = std::max(0.0, holosoma_prepare_hold_sec_);
    const double total_sec = ramp_sec + hold_sec;
    if (total_sec <= 1e-6) return;

    std::vector<float> target = model_to_motor(holosoma_initial_motion_pos_);
    if (holosoma_prepare_target_ == "stiff") target = model_to_motor(holosoma_stiff_startup_pos_);
    else if (holosoma_prepare_target_ == "default") target = model_to_motor(default_q_);
    const std::vector<float> prep_kp = model_to_motor(holosoma_stiff_startup_kp_);
    const std::vector<float> prep_kd = model_to_motor(holosoma_stiff_startup_kd_);
    holosoma_hold_q_ = target;
    holosoma_hold_kp_ = prep_kp;
    holosoma_hold_kd_ = prep_kd;
    const auto s0 = snapshot();
    std::vector<float> q0 = s0.q.size() >= motor_count_ ? s0.q : target;
    q0.resize(motor_count_, 0.f);
    std::cout << player_log_prefix() << " holosoma preparing target=" << holosoma_prepare_target_
              << " ramp=" << ramp_sec << " hold=" << hold_sec << " sec\n";
    while (!g_stop_requested.load()) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(now - start).count();
      if (elapsed >= total_sec) break;
      const double alpha = ramp_sec > 1e-6 ? std::min(elapsed / ramp_sec, 1.0) : 1.0;
      std::vector<float> q = lerp_vec(q0, target, alpha);
      const auto s = snapshot();
      publish(q, prep_kp, prep_kd, s.mode_machine);
      next += dt;
      if (next < std::chrono::steady_clock::now()) next = std::chrono::steady_clock::now();
      std::this_thread::sleep_until(next);
    }
    holosoma_hold_q_ = target;
    holosoma_hold_kp_ = prep_kp;
    holosoma_hold_kd_ = prep_kd;
    reset_holosoma_motion_buffers();
  }

void GenericPolicyPlayer::holosoma_capture_yaw_offsets() {
    const auto s = snapshot();
    holosoma_robot_yaw_offset_ = quat_yaw(torso_quat_from_state(s));
    holosoma_motion_yaw_offset_ = quat_yaw(xyzw_to_wxyz(holosoma_ref_quat_xyzw_));
    constexpr float kPi = 3.14159265358979323846f;
    std::cout << player_log_prefix() << " holosoma yaw offsets robot=" << (holosoma_robot_yaw_offset_ * 180.0f / kPi)
              << "deg motion=" << (holosoma_motion_yaw_offset_ * 180.0f / kPi) << "deg\n";
  }

void GenericPolicyPlayer::control_loop_holosoma() {
    const int publish_hz = std::max(inference_hz_, publish_hz_);
    const auto policy_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, inference_hz_)));
    const auto publish_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / publish_hz));
    const bool prestart_handoff = !start_file_.empty();
    if (prestart_handoff) {
      holosoma_set_current_pose_hold();
      const auto s = snapshot();
      publish(holosoma_hold_q_, holosoma_hold_kp_, holosoma_hold_kd_, s.mode_machine,
              false, CommandPhase::kHold);
      if (!ready_file_.empty()) {
        touch_file(ready_file_);
        std::cout << player_log_prefix() << " ready file touched: " << ready_file_ << "\n";
      }
      wait_start_file(holosoma_hold_q_, holosoma_hold_kp_, holosoma_hold_kd_, publish_hz);
      if (g_stop_requested.load()) return;
      holosoma_prepare_startup_pose(publish_hz);
    } else {
      holosoma_prepare_startup_pose(publish_hz);
      if (g_stop_requested.load()) return;
      if (!ready_file_.empty()) {
        touch_file(ready_file_);
        std::cout << player_log_prefix() << " ready file touched: " << ready_file_ << "\n";
      }
      wait_start_file(holosoma_hold_q_, holosoma_hold_kp_, holosoma_hold_kd_, publish_hz);
    }
    if (g_stop_requested.load()) return;
    holosoma_capture_yaw_offsets();
    holosoma_current_timestep_ = holosoma_motion_start_step_;
    holosoma_have_prev_policy_target_ = false;
    holosoma_prev_policy_target_q_.clear();
    holosoma_large_upper_body_tracking_warnings_ = 0;
    holosoma_large_target_step_warnings_ = 0;

    auto start_time = std::chrono::steady_clock::now();
    auto next_policy = start_time;
    auto next_publish = start_time;
    auto s = snapshot();
    std::vector<float> q = s.q.size() >= motor_count_ ? s.q : model_to_motor(holosoma_stiff_startup_pos_);
    q.resize(motor_count_, 0.f);
    std::vector<float> kp = model_to_motor(kp_);
    std::vector<float> kd = model_to_motor(kd_);
    std::vector<float> interp_from_q = q;
    std::vector<float> interp_to_q = q;
    auto interp_start = start_time;
    int64_t publish_count = 0;
    int64_t inference_count = 0;
    int64_t policy_deadline_miss = 0;
    int64_t publish_deadline_miss = 0;
    double avg_ms = 0.0, max_ms = 0.0;
    bool finished = false;
    bool motion_clip_progressing = holosoma_policy_warmup_sec_ <= 1e-6;
    holosoma_motion_clip_start_time_ = start_time;
    std::cout << player_log_prefix() << " start embedded_mimic_time/holosoma model=" << runtime_model_path_.string()
              << " motion_start_step=" << holosoma_motion_start_step_ << " motion_end_step=" << holosoma_motion_end_step_
              << " hz=" << inference_hz_ << " publish_hz=" << publish_hz
              << " policy_warmup_sec=" << holosoma_policy_warmup_sec_ << "\n";
    while (!g_stop_requested.load()) {
      const auto now = std::chrono::steady_clock::now();
      if (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start_time).count() >= duration_sec_) break;
      if (now >= next_policy) {
        policy_deadline_miss += advance_deadline(next_policy, policy_dt, now);
        s = snapshot();
        if (!motion_clip_progressing) {
          const double policy_elapsed = std::chrono::duration<double>(now - start_time).count();
          if (policy_elapsed >= holosoma_policy_warmup_sec_) {
            motion_clip_progressing = true;
            holosoma_current_timestep_ = holosoma_motion_start_step_;
            holosoma_motion_clip_start_time_ = now;
            std::cout << player_log_prefix() << " holosoma starting embedded motion clip at timestep "
                      << holosoma_motion_start_step_ << "\n";
          }
        }
        std::vector<float> next_q, next_kp, next_kd;
        const auto t0 = std::chrono::steady_clock::now();
        if (!build_holosoma_policy_command(s, next_q, next_kp, next_kd, motion_clip_progressing)) {
          finished = true;
        } else {
          const auto t1 = std::chrono::steady_clock::now();
          const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
          ++inference_count;
          avg_ms += (ms - avg_ms) / static_cast<double>(inference_count);
          max_ms = std::max(max_ms, ms);
          if (stats_interval_ > 0 && inference_count % stats_interval_ == 0) {
            std::cout << player_log_prefix() << " holosoma t=" << holosoma_current_timestep_
                      << " inference avg=" << avg_ms << "ms max=" << max_ms << "ms\n";
            event_log_.log("runtime_metrics",
                           {event_string("player", player_label()),
                            event_string("template", template_),
                            event_string("variant", variant_),
                            event_int("step", inference_count),
                            event_number("infer_ms_avg", avg_ms),
                            event_number("infer_ms_max", max_ms),
                            event_int("policy_deadline_miss", policy_deadline_miss),
                            event_int("publish_deadline_miss", publish_deadline_miss)});
            max_ms = 0.0;
          }
          interp_from_q = q;
          interp_to_q = std::move(next_q);
          interp_start = now;
          kp = std::move(next_kp);
          kd = std::move(next_kd);
        }
      }
      if (now >= next_publish) {
        publish_deadline_miss += advance_deadline(next_publish, publish_dt, now);
        if (target_interp_sec_ > 1e-6 && interp_from_q.size() == interp_to_q.size()) {
          q = lerp_vec(interp_from_q, interp_to_q,
                       std::chrono::duration<double>(now - interp_start).count() / target_interp_sec_);
        } else {
          q = interp_to_q;
        }
        s = snapshot();
        publish(q, kp, kd, s.mode_machine);
        ++publish_count;
      }
      if (finished) break;
      const auto sleep_until = std::min(next_policy, next_publish);
      if (sleep_until > std::chrono::steady_clock::now()) std::this_thread::sleep_until(sleep_until);
    }
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    std::cout << player_log_prefix() << " holosoma stats inference=" << inference_count
              << " publish=" << publish_count
              << " policy_deadline_miss=" << policy_deadline_miss
              << " publish_deadline_miss=" << publish_deadline_miss << "\n";
    event_log_.log("runtime_metrics",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_int("step", inference_count),
                    event_int("publish_count", publish_count),
                    event_int("policy_deadline_miss", policy_deadline_miss),
                    event_int("publish_deadline_miss", publish_deadline_miss),
                    event_string("phase", "final")});
    std::cout << player_log_prefix() << " completed holosoma\n";
  }

}  // namespace action_go::runtime::generic_policy_runtime
