#include "generic_policy_runtime_internal.h"

namespace action_go::runtime::generic_policy_runtime {

namespace {

using SuperController = action_go::runtime::SuperMimicController;

template <typename Values>
std::string supermimic_csv(const Values& values) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<float>::max_digits10);
  bool first = true;
  for (float value : values) {
    if (!first) stream << ',';
    first = false;
    stream << value;
  }
  return stream.str();
}

}

void GenericPolicyPlayer::load_spec_supermimic(const YAML::Node& y) {
  const auto reference = y["reference"];
  if (yamlc::required_scalar<std::string>(y, "reference.type") != "supermimic_csv_v1") {
    throw std::runtime_error("SuperMimic requires reference.type=supermimic_csv_v1");
  }
  const auto yaw = scalar_or<std::string>(reference, "yaw_alignment", "initial");
  if (yaw != "initial" && yaw != "none") {
    throw std::runtime_error("reference.yaw_alignment must be initial or none");
  }
  supermimic_align_initial_yaw_ = yaw == "initial";
  const auto end_behavior = scalar_or<std::string>(reference, "end_behavior", "hold");
  if (end_behavior != "hold" && end_behavior != "stop") {
    throw std::runtime_error("reference.end_behavior must be hold or stop");
  }
  supermimic_stop_at_end_ = end_behavior == "stop";
  const auto step_source = scalar_or<std::string>(reference, "policy_step_source", "wall_clock");
  if (step_source != "wall_clock" && step_source != "lowstate_tick") {
    throw std::runtime_error("reference.policy_step_source must be wall_clock or lowstate_tick");
  }
  supermimic_use_lowstate_tick_ = step_source == "lowstate_tick";
  supermimic_sim_lockstep_sequence_ = scalar_or<bool>(reference, "sim_lockstep_sequence", false);
  supermimic_trace_ = scalar_or<bool>(reference, "trace", false);
  if (inference_hz_ != 50 || scalar_or<int>(reference, "policy_tick_ms", 20) != 20) {
    throw std::runtime_error("SuperMimic requires inference_hz=50 and reference.policy_tick_ms=20");
  }
  if (supermimic_sim_lockstep_sequence_ && !supermimic_use_lowstate_tick_) {
    throw std::runtime_error("SuperMimic sim_lockstep_sequence requires policy_step_source=lowstate_tick");
  }
  if (sim_lowstate_decimation_ < 0) {
    throw std::runtime_error("--sim-lowstate-decimation must be nonnegative");
  }
  if (trajectory_path_.empty()) throw std::runtime_error("SuperMimic requires reference.path");
}

void GenericPolicyPlayer::load_supermimic_params() {
  const auto y = YAML::LoadFile(params_path_.string());
  num_actions_ = yamlc::required_scalar<int>(y, "policy.num_actions");
  num_obs_ = yamlc::required_scalar<int>(y, "policy.num_obs");
  if (motor_count_ != 29 || num_actions_ != 29 || num_obs_ != 1720 ||
      configured_action_dim_ != 29 || configured_obs_dim_ != 1720) {
    throw std::runtime_error("SuperMimic requires 29 motors/actions and 1720 packed observations");
  }
  joint_order_ = yamlc::required_vector_size<std::string>(y, "robot.policy_joint_order", motor_count_);
  joint_index_ = yamlc::required_vector_size<int>(y, "robot.joint_map", motor_count_);
  const auto& expected_names = SuperController::sdk_joint_names();
  for (int joint = 0; joint < motor_count_; ++joint) {
    if (joint_index_[joint] != joint || joint_order_[joint] != expected_names[joint]) {
      throw std::runtime_error("SuperMimic requires identity SDK joint_map and matching joint names at index " +
                               std::to_string(joint));
    }
  }
  default_q_ = yamlc::required_vector_size<float>(y, "robot.default_angles", motor_count_);
  action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", motor_count_);
  kp_ = yamlc::required_vector_size<float>(y, "robot.kp", motor_count_);
  kd_ = yamlc::required_vector_size<float>(y, "robot.kd", motor_count_);
  torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", motor_count_);
  lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", motor_count_);
  upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", motor_count_);
  for (int joint = 0; joint < motor_count_; ++joint) {
    if (!std::isfinite(kp_[joint]) || !std::isfinite(kd_[joint]) ||
        kp_[joint] < 0.f || kd_[joint] < 0.f ||
        !std::isfinite(torque_limits_[joint]) || torque_limits_[joint] <= 0.f) {
      throw std::runtime_error("SuperMimic requires finite nonnegative PD gains and positive torque limits");
    }
  }
  supermimic_controller_.configure(default_q_, action_scale_);
  supermimic_controller_.load_reference_csv(trajectory_path_.string());
  if (max_frames_ <= 0) max_frames_ = static_cast<int>(supermimic_controller_.reference_size());
  supermimic_controller_.set_frame_limit(static_cast<std::size_t>(max_frames_));
}

void GenericPolicyPlayer::post_load_supermimic() {
  if (obs_input_name_.empty()) obs_input_name_ = "obs";
  if (action_output_name_.empty()) action_output_name_ = "actions";
  onnx_.require_inputs({obs_input_name_});
  onnx_.require_outputs({action_output_name_});
  const auto& shapes = onnx_.input_shapes();
  if (shapes.size() != 1 || shapes[0].size() != 2 ||
      (shapes[0][0] != 1 && shapes[0][0] != -1) || shapes[0][1] != 1720) {
    throw std::runtime_error("SuperMimic ONNX input must be float obs [1,1720]");
  }
  supermimic_controller_.reset({1.f, 0.f, 0.f, 0.f}, false);
}

void GenericPolicyPlayer::schema_summary_supermimic() {
  schema_summary_common();
  std::cout << "  supermimic_contract packed_obs=1720 obs_frame=99 history=10 goal=730 actions=29"
            << " reference=" << trajectory_path_.string()
            << " yaw_alignment=" << (supermimic_align_initial_yaw_ ? "initial" : "none")
            << " end_behavior=" << (supermimic_stop_at_end_ ? "stop" : "hold")
            << " policy_step_source=" << (supermimic_use_lowstate_tick_ ? "lowstate_tick" : "wall_clock")
            << " policy_tick_ms=20 sim_lowstate_decimation=" << sim_lowstate_decimation_
            << " sim_lockstep_sequence=" << supermimic_sim_lockstep_sequence_ << "\n";
}

void GenericPolicyPlayer::check_only_supermimic() {
  SuperMimicRobotFrame robot;
  std::copy(default_q_.begin(), default_q_.end(), robot.q.begin());
  const auto input = supermimic_controller_.build_input(robot);
  const auto action = onnx_.run_single(input, obs_input_name_, action_output_name_, 29);
  supermimic_controller_.decode_action(action);
  supermimic_controller_.reset(robot.quat, false);
  std::cout << "  supermimic obs_dim=" << input.size() << " actions=" << action.size()
            << " reference_frames=" << supermimic_controller_.reference_size()
            << " max_frames=" << max_frames_ << " inference_check=passed\n";
}

void GenericPolicyPlayer::control_loop_supermimic() {
  using Clock = std::chrono::steady_clock;
  const int publish_hz = std::max(inference_hz_, publish_hz_);
  const auto policy_dt = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(0.02));
  const auto publish_dt = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(1.0 / publish_hz));
  auto require_fresh = [this](const RobotState& state) {
    if (!state.connected || state.q.size() < 29 || state.dq.size() < 29) {
      throw std::runtime_error("SuperMimic requires connected LowState with 29 joint positions and velocities");
    }
    if (command_safety_config_.lowstate_timeout_enabled &&
        std::chrono::duration<double>(Clock::now() - state.stamp).count() >
            command_safety_config_.lowstate_timeout_sec) {
      throw std::runtime_error("SuperMimic LowState is stale");
    }
  };
  auto robot_frame = [](const RobotState& state) {
    SuperMimicRobotFrame robot;
    std::copy_n(state.q.begin(), 29, robot.q.begin());
    std::copy_n(state.dq.begin(), 29, robot.dq.begin());
    robot.quat = state.quat;
    robot.gyro = state.gyro;
    return robot;
  };
  auto state = snapshot();
  require_fresh(state);
  std::vector<float> target, target_kp, target_kd;
  hold_current_pose(state, target, target_kp, target_kd);
  supermimic_command_sequence_.store(0);
  policy_counter_ = 0;
  publish(target, target_kp, target_kd, state.mode_machine, false, CommandPhase::kHold);
  if (!ready_file_.empty()) touch_file(ready_file_);

  bool preview_committed = false;
  uint32_t last_policy_tick_ms = state.source_tick_ms;
  uint64_t last_policy_state_seq = state.seq;
  auto infer = [&](const RobotState& policy_state) {
    require_fresh(policy_state);
    const auto reference_frame = supermimic_controller_.reference_index();
    const auto input = supermimic_controller_.build_input(robot_frame(policy_state));
    const auto inference_start = Clock::now();
    const auto action = onnx_.run_single(input, obs_input_name_, action_output_name_, 29);
    const double inference_ms = std::chrono::duration<double, std::milli>(Clock::now() - inference_start).count();
    auto next_target = supermimic_controller_.decode_action(action);
    supermimic_controller_.accept_action(action);
    target = std::move(next_target);
    target_kp = kp_;
    target_kd = kd_;
    policy_counter_ = static_cast<int64_t>(supermimic_controller_.policy_step());
    last_policy_tick_ms = policy_state.source_tick_ms;
    last_policy_state_seq = policy_state.seq;
    if (supermimic_trace_) {
      event_log_.log("supermimic_inference",
                     {event_string("player", player_label()), event_string("variant", variant_),
                      event_int("step", policy_counter_ - 1), event_int("reference_frame", reference_frame),
                      event_int("command_sequence", policy_counter_),
                      event_int("state_seq", static_cast<long long>(policy_state.seq)),
                      event_int("source_tick_ms", policy_state.source_tick_ms),
                      event_number("inference_ms", inference_ms),
                      event_string("q_csv", supermimic_csv(policy_state.q)),
                      event_string("dq_csv", supermimic_csv(policy_state.dq)),
                      event_string("quat_csv", supermimic_csv(policy_state.quat)),
                      event_string("gyro_csv", supermimic_csv(policy_state.gyro)),
                      event_string("obs_csv", supermimic_csv(input)),
                      event_string("actions_csv", supermimic_csv(action)),
                      event_string("target_csv", supermimic_csv(target))});
    }
  };

  if (preview_mode_enabled()) {
    supermimic_controller_.reset(state.quat, supermimic_align_initial_yaw_);
    const auto saved_controller = supermimic_controller_;
    infer(state);
    const auto decision = wait_preview_decision(target, target_kp, target_kd, state.mode_machine, publish_hz, 0);
    if (decision != PreviewDecision::kCommit) {
      supermimic_controller_ = saved_controller;
      policy_counter_ = 0;
      event_log_.log("preview_state_restored", {event_string("player", player_label()), event_string("variant", variant_)});
      if (decision == PreviewDecision::kTimeout) {
        throw std::runtime_error("transactional preview timed out before commit");
      }
      return;
    }
    preview_committed = true;
  } else {
    wait_start_file(target, target_kp, target_kd, publish_hz,
                    [&]() { require_fresh(snapshot()); }, true);
    if (g_stop_requested.load()) return;
    state = snapshot();
    require_fresh(state);
    supermimic_controller_.reset(state.quat, supermimic_align_initial_yaw_);
    hold_current_pose(state, target, target_kp, target_kd);
    last_policy_tick_ms = state.source_tick_ms;
    last_policy_state_seq = state.seq;
  }

  const auto start = Clock::now();
  auto next_policy = preview_committed ? start + policy_dt : start;
  auto next_publish = start;
  int64_t policy_deadline_miss = 0;
  int64_t sync_publish_count = 0;
  int64_t sync_publish_deadline_miss = 0;
  double sync_max_publish_interval_ms = 0.0;
  Clock::time_point previous_sync_publish{};
  auto synchronous_publish = [&]() {
    supermimic_command_sequence_.store(static_cast<uint32_t>(policy_counter_));
    publish(target, target_kp, target_kd, snapshot().mode_machine, false, CommandPhase::kActive);
    const auto now = Clock::now();
    if (previous_sync_publish != Clock::time_point{}) {
      sync_max_publish_interval_ms = std::max(sync_max_publish_interval_ms,
          std::chrono::duration<double, std::milli>(now - previous_sync_publish).count());
    }
    previous_sync_publish = now;
    ++sync_publish_count;
  };
  std::unique_ptr<AsyncCommandPublisher> async_publisher;
  if (supermimic_sim_lockstep_sequence_) {
    if (preview_committed) synchronous_publish();
  } else {
    async_publisher = std::make_unique<AsyncCommandPublisher>(
        publish_hz, 0.0, JointCommand{target, target_kp, target_kd},
        [this](const JointCommand& command, bool active) {
          const auto current_state = snapshot();
          publish(command.q, command.kp, command.kd, current_state.mode_machine, false,
                  active ? CommandPhase::kActive : CommandPhase::kHold);
        }, preview_committed);
  }
  std::cout << player_log_prefix() << " start motion_conditioned_tracking/supermimic"
            << " hz=50 publish_hz=" << publish_hz
            << " reference_frames=" << supermimic_controller_.frame_limit()
            << " end_behavior=" << (supermimic_stop_at_end_ ? "stop" : "hold")
            << " sim_lockstep_sequence=" << supermimic_sim_lockstep_sequence_ << "\n";
  std::exception_ptr inference_error;
  bool finished = false;
  try {
    while (!g_stop_requested.load() && (!async_publisher || !async_publisher->failed())) {
      const auto now = Clock::now();
      if (duration_sec_ > 0.0 && std::chrono::duration<double>(now - start).count() >= duration_sec_) {
        finished = true;
        break;
      }
      state = snapshot();
      require_fresh(state);
      const uint32_t tick_elapsed = state.source_tick_ms - last_policy_tick_ms;
      const bool first_step = policy_counter_ == 0;
      const bool states_due = first_step || sim_lowstate_decimation_ <= 0 ||
          state.seq - last_policy_state_seq >= static_cast<uint64_t>(sim_lowstate_decimation_);
      const bool policy_due = states_due && (first_step ||
          (supermimic_use_lowstate_tick_ ? tick_elapsed >= SuperController::kPolicyTickMs : now >= next_policy));
      if (policy_due) {
        if (supermimic_stop_at_end_ && supermimic_controller_.at_end()) {
          finished = true;
          break;
        }
        if (!first_step && supermimic_use_lowstate_tick_ && tick_elapsed >= 2 * SuperController::kPolicyTickMs) {
          policy_deadline_miss += tick_elapsed / SuperController::kPolicyTickMs - 1;
        }
        if (!supermimic_use_lowstate_tick_) {
          policy_deadline_miss += advance_deadline(next_policy, policy_dt, now);
        }
        infer(state);
        if (supermimic_sim_lockstep_sequence_) {
          synchronous_publish();
          next_publish = Clock::now() + publish_dt;
        } else {
          async_publisher->update({target, target_kp, target_kd});
        }
        if (stats_interval_ > 0 && policy_counter_ % stats_interval_ == 0) {
          event_log_.log("runtime_metrics",
                         {event_string("player", player_label()), event_string("variant", variant_),
                          event_int("step", policy_counter_),
                          event_int("reference_frame", supermimic_controller_.reference_index()),
                          event_int("policy_deadline_miss", policy_deadline_miss)});
        }
      }
      if (supermimic_sim_lockstep_sequence_ && policy_counter_ > 0 && Clock::now() >= next_publish) {
        sync_publish_deadline_miss += advance_deadline(next_publish, publish_dt, Clock::now());
        synchronous_publish();
      }
      if (supermimic_sim_lockstep_sequence_) {
        std::this_thread::sleep_until(std::min(next_publish, Clock::now() + std::chrono::milliseconds(1)));
      } else if (supermimic_use_lowstate_tick_ || sim_lowstate_decimation_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else if (next_policy > Clock::now()) {
        std::this_thread::sleep_until(next_policy);
      }
    }
  } catch (...) {
    inference_error = std::current_exception();
  }
  if (async_publisher) {
    async_publisher->stop_and_join();
    async_publisher->rethrow_if_failed();
  }
  if (inference_error) std::rethrow_exception(inference_error);
  if (finished && !g_stop_requested.load() && !handoff_complete_file_.empty()) {
    if (last_published_q_.size() == static_cast<std::size_t>(motor_count_)) {
      target = last_published_q_;
      target_kp = last_published_kp_;
      target_kd = last_published_kd_;
    }
    event_log_.log("handoff_hold_started",
                   {event_string("player", player_label()), event_string("variant", variant_),
                    event_string("complete_file", handoff_complete_file_),
                    event_string("release_file", handoff_release_file_)});
    publish(target, target_kp, target_kd, snapshot().mode_machine, false, CommandPhase::kHold);
    touch_file(handoff_complete_file_);
    auto next_hold = Clock::now() + publish_dt;
    while (!g_stop_requested.load() && !fs::exists(handoff_release_file_)) {
      std::this_thread::sleep_until(next_hold);
      if (g_stop_requested.load() || fs::exists(handoff_release_file_)) break;
      publish(target, target_kp, target_kd, snapshot().mode_machine, false, CommandPhase::kHold);
      next_hold = Clock::now() + publish_dt;
    }
    event_log_.log("handoff_hold_stopped",
                   {event_string("player", player_label()), event_string("variant", variant_),
                    event_bool("released", fs::exists(handoff_release_file_))});
  }
  if (g_stop_requested.load() && damping_on_stop_) publish_damping_for_stop();
  const auto metrics = async_publisher ? async_publisher->metrics() :
      AsyncPublisherMetrics{sync_publish_count, sync_publish_deadline_miss, sync_max_publish_interval_ms};
  event_log_.log("runtime_metrics",
                 {event_string("player", player_label()), event_string("variant", variant_),
                  event_int("step", policy_counter_), event_int("publish_count", metrics.publish_count),
                  event_int("policy_deadline_miss", policy_deadline_miss),
                  event_int("publish_deadline_miss", metrics.deadline_miss),
                  event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                  event_string("phase", "final")});
}

}
