#include "generic_policy_runtime_internal.h"

namespace action_go::runtime::generic_policy_runtime {

void GenericPolicyPlayer::load_spec_bfm(const YAML::Node& y) {
    const YAML::Node runtime = y["runtime"];
    const YAML::Node obs = y["observation"];
    const YAML::Node action = y["action"];
    bfm_actor_obs_dim_ = configured_obs_dim_;
    bfm_z_dim_ = yamlc::required_scalar<int>(y, "observation.latent_dim");
    bfm_history_steps_ = yamlc::required_scalar<int>(y, "observation.history_steps");
    if (bfm_z_dim_ <= 0 || bfm_history_steps_ < 0) {
      throw std::runtime_error("BFM observation latent_dim/history_steps must be non-negative");
    }
    if (bfm_z_dim_ != protocol_->latent_dim || bfm_history_steps_ != protocol_->history_steps) {
      throw std::runtime_error("BFM-Zero observation latent_dim/history_steps do not match protocol constraint");
    }
    latent_file_ = scalar_or<std::string>(obs, "latent_file", scalar_or<std::string>(y["model"], "latent_file", ""));
    bfm_latent_sequence_file_ = scalar_or<std::string>(obs, "latent_sequence_file", "");
    bfm_latent_sequence_loop_ = scalar_or<bool>(obs, "latent_sequence_loop", true);
    if (!args_.latent_file_override.empty()) {
      latent_file_ = args_.latent_file_override;
      bfm_latent_sequence_file_.clear();
    }
    bfm_warmup_steps_ = scalar_or<int>(runtime, "warmup_steps", 5);
    bfm_action_rescale_override_ = scalar_or<float>(action, "action_rescale", std::numeric_limits<float>::quiet_NaN());
  }

void GenericPolicyPlayer::load_spec_unitree(const YAML::Node& y) {
    const YAML::Node runtime = y["runtime"];
    const YAML::Node obs = y["observation"];
    const YAML::Node action = y["action"];
    unitree_base_obs_dim_ = yamlc::required_scalar<int>(y, "observation.base_obs_dim");
    unitree_obs_dim_ = configured_obs_dim_;
    unitree_allowed_obs_dims_ = yamlc::required_vector<int>(y, "observation.allowed_obs_dims");
    if (unitree_allowed_obs_dims_.empty()) throw std::runtime_error("observation.allowed_obs_dims must not be empty");
    if (unitree_base_obs_dim_ != protocol_->base_obs_dim ||
        unitree_allowed_obs_dims_ != protocol_->allowed_obs_dims) {
      throw std::runtime_error("unitree_rl_mjlab observation.base_obs_dim/allowed_obs_dims do not match protocol constraint");
    }
    unitree_command_source_ = lower(scalar_or<std::string>(runtime, "command_source", scalar_or<std::string>(obs, "command_source", "remote")));
    unitree_lin_deadband_ = scalar_or<float>(runtime, "lin_vel_deadband", 0.10f);
    unitree_ang_deadband_ = scalar_or<float>(runtime, "ang_vel_deadband", 0.10f);
    unitree_max_action_abs_ = scalar_or<float>(action, "max_action_abs", 5.0f);
    unitree_clip_by_torque_ = scalar_or<bool>(action, "clip_by_torque", true);
    unitree_startup_sec_ = scalar_or<double>(runtime, "startup_sec", 0.2);
    unitree_kp_factor_ = scalar_or<float>(action, "kp_factor", 1.0f);
    unitree_kd_factor_ = scalar_or<float>(action, "kd_factor", 1.0f);
    unitree_max_vx_ = scalar_or<float>(runtime, "max_vx", std::numeric_limits<float>::quiet_NaN());
    unitree_max_back_vx_ = scalar_or<float>(runtime, "max_back_vx", std::numeric_limits<float>::quiet_NaN());
    unitree_max_vy_ = scalar_or<float>(runtime, "max_vy", std::numeric_limits<float>::quiet_NaN());
    unitree_max_yaw_ = scalar_or<float>(runtime, "max_yaw", std::numeric_limits<float>::quiet_NaN());
  }

void GenericPolicyPlayer::post_load_bfm() {
    obs_input_name_ = obs_input_name_.empty() ? "actor_obs" : obs_input_name_;
    action_output_name_ = action_output_name_.empty() ? "action" : action_output_name_;
    onnx_.require_inputs({obs_input_name_});
    onnx_.require_outputs({action_output_name_});
  }

void GenericPolicyPlayer::post_load_unitree() {
    unitree_obs_dim_ = onnx_.inferred_obs_dim(unitree_obs_dim_);
    if (std::find(unitree_allowed_obs_dims_.begin(), unitree_allowed_obs_dims_.end(), unitree_obs_dim_) ==
        unitree_allowed_obs_dims_.end()) {
      std::ostringstream allowed;
      for (size_t i = 0; i < unitree_allowed_obs_dims_.size(); ++i) {
        if (i) allowed << ", ";
        allowed << unitree_allowed_obs_dims_[i];
      }
      throw std::runtime_error("unsupported unitree_rl_mjlab ONNX obs dim=" + std::to_string(unitree_obs_dim_) +
                               " (allowed by spec: " + allowed.str() + ")");
    }
    obs_input_name_ = obs_input_name_.empty() ? "obs" : obs_input_name_;
    action_output_name_ = action_output_name_.empty() ? "actions" : action_output_name_;
    onnx_.require_inputs({obs_input_name_});
    onnx_.require_outputs({action_output_name_});
  }

void GenericPolicyPlayer::load_bfm_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("BFM-Zero params not found: " + params_path_.string());
    const YAML::Node y = YAML::LoadFile(params_path_.string());
    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.joint_order", motor_count_);
    default_q_ = yamlc::required_vector_size<float>(y, "robot.default_angles", motor_count_);
    action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", motor_count_);
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
    bfm_action_rescale_ = std::isfinite(bfm_action_rescale_override_)
                              ? bfm_action_rescale_override_
                              : yamlc::required_scalar<float>(y, "action.action_rescale");
    const int params_actor_obs_dim = yamlc::required_scalar<int>(y, "policy.actor_obs_dim");
    if (configured_obs_dim_ > 0 && params_actor_obs_dim != configured_obs_dim_) {
      throw std::runtime_error("BFM-Zero params policy.actor_obs_dim does not match spec observation.obs_dim");
    }
    bfm_actor_obs_dim_ = params_actor_obs_dim;

    bfm_latent_sequence_.clear();
    bfm_latent_sequence_frames_ = 0;
    bfm_latent_sequence_index_ = 0;
    if (!bfm_latent_sequence_file_.empty()) {
      const fs::path sequence_path = resolve_relative(params_path_, bfm_latent_sequence_file_);
      const auto sequence_bytes = read_binary_file(sequence_path);
      const auto sequence_meta = action_go::runtime::npy::parse_header(sequence_bytes);
      if (sequence_meta.shape.size() != 2 || sequence_meta.shape[0] == 0 ||
          sequence_meta.shape[1] != static_cast<size_t>(bfm_z_dim_)) {
        throw std::runtime_error("BFM latent sequence must have shape [frames," +
                                 std::to_string(bfm_z_dim_) + "]: " + sequence_path.string());
      }
      bfm_latent_sequence_ = npy_to_float_vector(sequence_bytes, sequence_path.string());
      bfm_latent_sequence_frames_ = sequence_meta.shape[0];
      bfm_z_.assign(bfm_latent_sequence_.begin(),
                    bfm_latent_sequence_.begin() + bfm_z_dim_);
    } else {
      std::string z_path = latent_file_;
      if (z_path.empty()) z_path = yamlc::required_scalar<std::string>(y, "observation.z_file");
      if (z_path.empty()) throw std::runtime_error("BFM-Zero requires observation.latent_file, latent_sequence_file, or params z_file");
      bfm_z_ = read_float_file(resolve_relative(params_path_, z_path), static_cast<size_t>(bfm_z_dim_));
    }
    last_action_ = std::vector<float>(motor_count_, 0.f);
    bfm_last_action_scaled_.assign(motor_count_, 0.f);
    bfm_hist_prev_actions_.assign(bfm_history_steps_ * motor_count_, 0.f);
    bfm_hist_base_ang_vel_.assign(bfm_history_steps_ * 3, 0.f);
    bfm_hist_dof_pos_.assign(bfm_history_steps_ * motor_count_, 0.f);
    bfm_hist_dof_vel_.assign(bfm_history_steps_ * motor_count_, 0.f);
    bfm_hist_projected_gravity_.assign(bfm_history_steps_ * 3, 0.f);
  }

void GenericPolicyPlayer::load_unitree_params() {
    if (!fs::exists(params_path_)) throw std::runtime_error("unitree_rl_mjlab params not found: " + params_path_.string());
    const YAML::Node y = YAML::LoadFile(params_path_.string());
    joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.joint_order", motor_count_);
    default_q_ = yamlc::required_vector_size<float>(y, "robot.default_angles", motor_count_);
    action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", motor_count_);
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
    for (float& v : kp_) v *= unitree_kp_factor_;
    for (float& v : kd_) v *= unitree_kd_factor_;
    unitree_default_max_vx_ = yamlc::required_scalar<float>(y, "motion.max_vx");
    unitree_default_back_vx_ = yamlc::required_scalar<float>(y, "motion.max_back_vx");
    unitree_default_max_vy_ = yamlc::required_scalar<float>(y, "motion.max_vy");
    unitree_default_max_yaw_ = yamlc::required_scalar<float>(y, "motion.max_yaw");
    unitree_phase_period_ = yamlc::required_scalar<double>(y, "observation.phase_period");
    last_action_.assign(motor_count_, 0.f);
  }

void GenericPolicyPlayer::schema_summary_bfm() {
    schema_summary_common();
    std::cout << "  bfm_contract latent_dim=" << bfm_z_dim_
              << " history_steps=" << bfm_history_steps_ << "\n";
  }

void GenericPolicyPlayer::schema_summary_unitree() {
    schema_summary_common();
    std::cout << "  unitree_contract base_obs_dim=" << unitree_base_obs_dim_
              << " allowed_obs_dims=[" << join_ints(unitree_allowed_obs_dims_) << "]\n";
  }

void GenericPolicyPlayer::check_only_bfm() {
    std::vector<float> obs(static_cast<size_t>(bfm_actor_obs_dim_), 0.f);
    auto action = onnx_.run_single(obs, obs_input_name_, action_output_name_, motor_count_);
    std::cout << "  bfm_zero obs_dim=" << bfm_actor_obs_dim_ << " z_dim=" << bfm_z_.size()
              << " action_dim=" << action.size();
    if (bfm_latent_sequence_frames_ > 0) {
      std::cout << " latent_sequence_frames=" << bfm_latent_sequence_frames_
                << " loop=" << (bfm_latent_sequence_loop_ ? "true" : "false");
    }
    std::cout << "\n";
  }

void GenericPolicyPlayer::check_only_unitree() {
    std::vector<float> obs(static_cast<size_t>(unitree_obs_dim_), 0.f);
    auto action = onnx_.run_single(obs, obs_input_name_, action_output_name_, motor_count_);
    std::cout << "  unitree_rl_mjlab obs_dim=" << unitree_obs_dim_ << " action_dim=" << action.size()
              << " phase_period=" << unitree_phase_period_ << "\n";
  }

std::vector<float> GenericPolicyPlayer::bfm_infer(const std::vector<float>& obs) {
    check_size("bfm actor_obs", static_cast<size_t>(bfm_actor_obs_dim_), obs.size());
    return onnx_.run_single(obs, obs_input_name_, action_output_name_, motor_count_);
  }

bool GenericPolicyPlayer::build_bfm_target(const GenericPolicyPlayer::RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                        std::vector<float>& action_out) {
    if (s.q.size() < motor_count_ || s.dq.size() < motor_count_) throw std::runtime_error("lowstate motor count too small");
    std::vector<float> dof_pos(motor_count_), dof_vel(motor_count_), base_ang_vel_scaled(3);
    for (int i = 0; i < motor_count_; ++i) {
      dof_pos[i] = s.q[i] - default_q_[i];
      dof_vel[i] = s.dq[i];
    }
    for (int i = 0; i < 3; ++i) base_ang_vel_scaled[i] = s.gyro[i] * 0.25f;
    const auto projected_gravity_arr = projected_gravity_from_quat(s.quat);
    const std::vector<float> projected_gravity(projected_gravity_arr.begin(), projected_gravity_arr.end());

    push_history(bfm_hist_prev_actions_, bfm_last_action_scaled_, motor_count_, bfm_history_steps_);
    push_history(bfm_hist_base_ang_vel_, base_ang_vel_scaled, 3, bfm_history_steps_);
    push_history(bfm_hist_dof_pos_, dof_pos, motor_count_, bfm_history_steps_);
    push_history(bfm_hist_dof_vel_, dof_vel, motor_count_, bfm_history_steps_);
    push_history(bfm_hist_projected_gravity_, projected_gravity, 3, bfm_history_steps_);

    if (policy_counter_ < bfm_warmup_steps_) {
      hold_current_pose(s, q, kp, kd);
      action_out.assign(motor_count_, 0.f);
      ++policy_counter_;
      return true;
    }

    if (bfm_latent_sequence_frames_ > 0) {
      const size_t offset = bfm_latent_sequence_index_ * static_cast<size_t>(bfm_z_dim_);
      std::copy_n(bfm_latent_sequence_.begin() + static_cast<std::ptrdiff_t>(offset),
                  bfm_z_dim_, bfm_z_.begin());
    }

    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(bfm_actor_obs_dim_));
    append(obs, dof_pos);
    append(obs, dof_vel);
    append(obs, projected_gravity_arr);
    append(obs, base_ang_vel_scaled);
    append(obs, bfm_last_action_scaled_);
    append(obs, bfm_hist_prev_actions_);
    append(obs, bfm_hist_base_ang_vel_);
    append(obs, bfm_hist_dof_pos_);
    append(obs, bfm_hist_dof_vel_);
    append(obs, bfm_hist_projected_gravity_);
    append(obs, bfm_z_);
    auto action = bfm_infer(obs);
    action_out = action;
    q.assign(motor_count_, 0.f);
    kp = kp_;
    kd = kd_;
    for (int i = 0; i < motor_count_; ++i) {
      const float clipped = std::clamp(action[i], -1.f, 1.f);
      bfm_last_action_scaled_[i] = bfm_action_rescale_ * clipped;
      q[i] = default_q_[i] + bfm_last_action_scaled_[i] * action_scale_[i];
      q[i] = std::clamp(q[i], lower_limits_[i], upper_limits_[i]);
    }
    if (bfm_latent_sequence_frames_ > 0) {
      if (bfm_latent_sequence_index_ + 1 < bfm_latent_sequence_frames_) {
        ++bfm_latent_sequence_index_;
      } else if (bfm_latent_sequence_loop_) {
        bfm_latent_sequence_index_ = 0;
      }
    }
    ++policy_counter_;
    return true;
  }

void GenericPolicyPlayer::control_loop_single_bfm() {
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

    policy_counter_ = 0;
    bfm_latent_sequence_index_ = 0;
    bool preview_committed = false;
    std::vector<float> last_action(motor_count_, 0.f);
    if (preview_mode_enabled()) {
      const auto saved_policy_counter = policy_counter_;
      const auto saved_latent_index = bfm_latent_sequence_index_;
      const auto saved_z = bfm_z_;
      const auto saved_last_action_scaled = bfm_last_action_scaled_;
      const auto saved_hist_prev_actions = bfm_hist_prev_actions_;
      const auto saved_hist_base_ang_vel = bfm_hist_base_ang_vel_;
      const auto saved_hist_dof_pos = bfm_hist_dof_pos_;
      const auto saved_hist_dof_vel = bfm_hist_dof_vel_;
      const auto saved_hist_projected_gravity = bfm_hist_projected_gravity_;
      auto restore_preview_state = [&]() {
        policy_counter_ = saved_policy_counter;
        bfm_latent_sequence_index_ = saved_latent_index;
        bfm_z_ = saved_z;
        bfm_last_action_scaled_ = saved_last_action_scaled;
        bfm_hist_prev_actions_ = saved_hist_prev_actions;
        bfm_hist_base_ang_vel_ = saved_hist_base_ang_vel;
        bfm_hist_dof_pos_ = saved_hist_dof_pos;
        bfm_hist_dof_vel_ = saved_hist_dof_vel;
        bfm_hist_projected_gravity_ = saved_hist_projected_gravity;
      };

      for (int step = 0; step <= bfm_warmup_steps_; ++step) {
        build_bfm_target(s, q, kp, kd, last_action);
      }
      const auto decision = wait_preview_decision(
          q, kp, kd, s.mode_machine, publish_hz, bfm_warmup_steps_);
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
      wait_start_file(q, kp, kd, publish_hz);
      if (g_stop_requested.load()) return;
    }

    auto start = std::chrono::steady_clock::now();
    auto next_policy = preview_committed ? start + policy_dt : start;
    auto next_publish = start;
    std::vector<float> target_begin = q, target_goal = q, filtered_target = q;
    auto target_update_time = start;
    int64_t steps = 0;
    double avg_ms = 0.0, max_ms = 0.0;
    std::cout << player_log_prefix() << " start single_actor_state/bfm_zero model=" << runtime_model_path_.string()
              << " hz=" << inference_hz_ << " publish_hz=" << publish_hz
              << " target_interp_sec=" << target_interp_sec_ << "\n";
    while (!g_stop_requested.load()) {
      const auto now = std::chrono::steady_clock::now();
      if (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start).count() >= duration_sec_) break;
      if (now >= next_policy) {
        while (next_policy <= now) next_policy += policy_dt;
        s = snapshot();
        std::vector<float> new_goal, new_kp, new_kd;
        const auto t0 = std::chrono::steady_clock::now();
        build_bfm_target(s, new_goal, new_kp, new_kd, last_action);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++steps;
        avg_ms += (ms - avg_ms) / static_cast<double>(steps);
        max_ms = std::max(max_ms, ms);
        target_begin = filtered_target;
        target_goal = std::move(new_goal);
        kp = std::move(new_kp);
        kd = std::move(new_kd);
        target_update_time = now;
        if (stats_interval_ > 0 && steps % stats_interval_ == 0) {
          float action_abs_max = 0.f;
          for (float v : last_action) action_abs_max = std::max(action_abs_max, std::abs(v));
          std::cout << player_log_prefix() << " bfm step=" << steps << " inference avg=" << avg_ms
                    << "ms max=" << max_ms << "ms action_abs_max=" << action_abs_max;
          if (bfm_latent_sequence_frames_ > 0) {
            std::cout << " latent_frame=" << bfm_latent_sequence_index_ << "/"
                      << bfm_latent_sequence_frames_;
          }
          std::cout << "\n";
          event_log_.log("runtime_metrics",
                         {event_string("player", player_label()),
                          event_string("template", template_),
                          event_string("variant", variant_),
                          event_int("step", steps),
                          event_number("infer_ms_avg", avg_ms),
                          event_number("infer_ms_max", max_ms),
                          event_number("action_abs_max", action_abs_max)});
          max_ms = 0.0;
        }
      }
      if (now >= next_publish) {
        while (next_publish <= now) next_publish += publish_dt;
        if (target_interp_sec_ > 1e-6 && target_begin.size() == target_goal.size()) {
          filtered_target = lerp_vec(target_begin, target_goal,
                                     std::chrono::duration<double>(now - target_update_time).count() / target_interp_sec_);
        } else {
          filtered_target = target_goal;
        }
        s = snapshot();
        publish(filtered_target, kp, kd, s.mode_machine);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    std::cout << player_log_prefix() << " completed bfm_zero\n";
  }

GenericPolicyPlayer::Command GenericPolicyPlayer::unitree_command_from_remote(const std::array<uint8_t, 40>& remote) const {
    if (unitree_command_source_ == "zero") return {};
    const float lx_raw = sanitize_axis(unpack_float_le(remote, 4));
    const float rx_raw = sanitize_axis(unpack_float_le(remote, 8));
    const float ly_raw = sanitize_axis(unpack_float_le(remote, 20));
    const float max_vx = std::isfinite(unitree_max_vx_) ? unitree_max_vx_ : unitree_default_max_vx_;
    const float back_vx = std::isfinite(unitree_max_back_vx_) ? unitree_max_back_vx_ : unitree_default_back_vx_;
    const float max_vy = std::isfinite(unitree_max_vy_) ? unitree_max_vy_ : unitree_default_max_vy_;
    const float max_yaw = std::isfinite(unitree_max_yaw_) ? unitree_max_yaw_ : unitree_default_max_yaw_;
    Command cmd;
    cmd.vx = apply_deadband(ly_raw, unitree_lin_deadband_, max_vx, -std::abs(back_vx));
    cmd.vy = apply_deadband(-lx_raw, unitree_lin_deadband_, max_vy, -std::abs(max_vy));
    cmd.yaw = apply_deadband(-rx_raw, unitree_ang_deadband_, max_yaw, -std::abs(max_yaw));
    return cmd;
  }

std::vector<float> GenericPolicyPlayer::build_unitree_obs(const GenericPolicyPlayer::RobotState& s, const GenericPolicyPlayer::Command& c) {
    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(unitree_obs_dim_));
    for (float v : s.gyro) obs.push_back(v);
    const auto g = projected_gravity_from_quat(s.quat);
    for (float v : g) obs.push_back(v);
    obs.push_back(c.vx); obs.push_back(c.vy); obs.push_back(c.yaw);
    if (unitree_obs_dim_ == unitree_base_obs_dim_ + 2) {
      const float command_norm = std::sqrt(c.vx * c.vx + c.vy * c.vy + c.yaw * c.yaw);
      if (command_norm < 0.1f) {
        obs.push_back(0.f); obs.push_back(0.f);
      } else {
        const double period = std::max(1e-6, unitree_phase_period_);
        const double p = std::fmod(std::max(0.0, unitree_phase_time_), period) / period;
        obs.push_back(static_cast<float>(std::sin(p * 2.0 * M_PI)));
        obs.push_back(static_cast<float>(std::cos(p * 2.0 * M_PI)));
      }
      unitree_phase_time_ += 1.0 / std::max(1, inference_hz_);
    }
    for (int i = 0; i < motor_count_; ++i) obs.push_back(s.q[i] - default_q_[i]);
    for (int i = 0; i < motor_count_; ++i) obs.push_back(s.dq[i]);
    for (int i = 0; i < motor_count_; ++i) obs.push_back(last_action_[i]);
    if (unitree_obs_dim_ == unitree_base_obs_dim_ + 1) obs.push_back(0.f);
    check_size("unitree obs", static_cast<size_t>(unitree_obs_dim_), obs.size());
    return obs;
  }

bool GenericPolicyPlayer::build_unitree_target(const GenericPolicyPlayer::RobotState& s, std::vector<float>& target, std::vector<float>& kp, std::vector<float>& kd,
                            GenericPolicyPlayer::Command& cmd, std::vector<float>& action) {
    cmd = unitree_command_from_remote(s.wireless_remote);
    auto obs = build_unitree_obs(s, cmd);
    action = onnx_.run_single(obs, obs_input_name_, action_output_name_, motor_count_);
    if (unitree_max_action_abs_ > 0.f) {
      for (float& v : action) v = std::clamp(v, -unitree_max_action_abs_, unitree_max_action_abs_);
    }
    last_action_ = action;
    target.assign(motor_count_, 0.f);
    kp = kp_;
    kd = kd_;
    for (int i = 0; i < motor_count_; ++i) {
      float q = default_q_[i] + action[i] * action_scale_[i];
      q = std::clamp(q, lower_limits_[i], upper_limits_[i]);
      if (unitree_clip_by_torque_) {
        const float kp_safe = std::max(kp_[i], 1e-6f);
        const float low = (-torque_limits_[i] + kd_[i] * s.dq[i]) / kp_safe + s.q[i];
        const float high = (torque_limits_[i] + kd_[i] * s.dq[i]) / kp_safe + s.q[i];
        q = std::clamp(q, low, high);
      }
      target[i] = q;
    }
    return true;
  }

void GenericPolicyPlayer::control_loop_single_unitree() {
    auto s = snapshot();
    std::vector<float> q = s.q;
    q.resize(motor_count_, 0.f);
    std::vector<float> kp = kp_, kd = kd_;
    publish(q, kp, kd, s.mode_machine, false, CommandPhase::kHold);
    if (!ready_file_.empty()) {
      touch_file(ready_file_);
      std::cout << player_log_prefix() << " ready file touched: " << ready_file_ << "\n";
    }
    wait_start_file(q, kp, kd, std::max(inference_hz_, publish_hz_));
    if (g_stop_requested.load()) return;

    const auto start = std::chrono::steady_clock::now();
    auto next_policy = start;
    auto next_publish = start;
    const auto policy_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, inference_hz_)));
    const auto publish_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, publish_hz_)));
    std::vector<float> target_begin = q, target_goal = q, filtered_target = q;
    auto target_update_time = start;
    Command last_cmd;
    std::vector<float> action(motor_count_, 0.f);
    int64_t steps = 0;
    double avg_ms = 0.0;
    double max_ms = 0.0;
    unitree_phase_time_ = 0.0;
    std::cout << player_log_prefix() << " start single_actor_state/unitree_rl_mjlab model=" << runtime_model_path_.string()
              << " obs_dim=" << unitree_obs_dim_ << " hz=" << inference_hz_ << " publish_hz=" << publish_hz_
              << " command_source=" << unitree_command_source_ << "\n";
    while (!g_stop_requested.load()) {
      const auto now = std::chrono::steady_clock::now();
      if (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start).count() >= duration_sec_) break;
      if (now >= next_policy) {
        while (next_policy <= now) next_policy += policy_dt;
        s = snapshot();
        std::vector<float> new_goal;
        if (unitree_startup_sec_ > 1e-6 && std::chrono::duration<double>(now - start).count() < unitree_startup_sec_) {
          const double alpha = std::clamp(std::chrono::duration<double>(now - start).count() / unitree_startup_sec_, 0.0, 1.0);
          new_goal = lerp_vec(q, default_q_, alpha);
          last_cmd = unitree_command_from_remote(s.wireless_remote);
          action.assign(motor_count_, 0.f);
        } else {
          const auto t0 = std::chrono::steady_clock::now();
          build_unitree_target(s, new_goal, kp, kd, last_cmd, action);
          const auto t1 = std::chrono::steady_clock::now();
          const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
          avg_ms += (ms - avg_ms) / static_cast<double>(steps + 1);
          max_ms = std::max(max_ms, ms);
        }
        target_begin = filtered_target;
        target_goal = std::move(new_goal);
        target_update_time = now;
        ++steps;
        if (stats_interval_ > 0 && steps % stats_interval_ == 0) {
          float action_abs_max = 0.f;
          for (float v : action) action_abs_max = std::max(action_abs_max, std::abs(v));
          std::cout << player_log_prefix() << " unitree step=" << steps
                    << " cmd=(" << last_cmd.vx << "," << last_cmd.vy << "," << last_cmd.yaw << ")"
                    << " tilt=" << tilt_deg_from_quat(s.quat) << "deg action_abs_max=" << action_abs_max << "\n";
          event_log_.log("runtime_metrics",
                         {event_string("player", player_label()),
                          event_string("template", template_),
                          event_string("variant", variant_),
                          event_int("step", steps),
                          event_number("infer_ms_avg", avg_ms),
                          event_number("infer_ms_max", max_ms),
                          event_number("action_abs_max", action_abs_max),
                          event_number("cmd_vx", last_cmd.vx),
                          event_number("cmd_vy", last_cmd.vy),
                          event_number("cmd_yaw", last_cmd.yaw)});
          max_ms = 0.0;
        }
      }
      if (now >= next_publish) {
        while (next_publish <= now) next_publish += publish_dt;
        if (target_interp_sec_ > 1e-6) {
          filtered_target = lerp_vec(target_begin, target_goal,
                                     std::chrono::duration<double>(now - target_update_time).count() / target_interp_sec_);
        } else {
          filtered_target = target_goal;
        }
        s = snapshot();
        publish(filtered_target, kp, kd, s.mode_machine);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
    std::cout << player_log_prefix() << " completed unitree_rl_mjlab\n";
  }

}  // namespace action_go::runtime::generic_policy_runtime
