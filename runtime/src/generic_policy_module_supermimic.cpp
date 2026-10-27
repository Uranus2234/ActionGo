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
  const auto expected_reference = variant_ == "allmimic" ? "allmimic_csv_v1" : "supermimic_csv_v1";
  if (yamlc::required_scalar<std::string>(y, "reference.type") != expected_reference) {
    throw std::runtime_error(variant_ + " requires reference.type=" + expected_reference);
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
  if (motor_count_ != 29 || num_actions_ != 29 || num_obs_ != (variant_ == "allmimic" ? 1660 : 1720) ||
      configured_action_dim_ != 29 || configured_obs_dim_ != num_obs_) {
    throw std::runtime_error(variant_ + " has incompatible motor/action/observation dimensions");
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
  if (variant_ == "allmimic") {
    safety_limits_.torque = torque_limits_;
    allmimic_controller_.configure(default_q_, action_scale_,
        yamlc::required_vector_size<float>(y, "action.residual_scale", motor_count_),
        yamlc::required_vector_size<float>(y, "action.residual_reference", motor_count_));
    allmimic_controller_.load_reference_csv(trajectory_path_.string());
    if (max_frames_ <= 0) max_frames_ = static_cast<int>(allmimic_controller_.reference_size());
    allmimic_controller_.set_frame_limit(static_cast<std::size_t>(max_frames_));
  } else {
    supermimic_controller_.configure(default_q_, action_scale_);
    supermimic_controller_.load_reference_csv(trajectory_path_.string());
    if (max_frames_ <= 0) max_frames_ = static_cast<int>(supermimic_controller_.reference_size());
    supermimic_controller_.set_frame_limit(static_cast<std::size_t>(max_frames_));
  }
}

void GenericPolicyPlayer::post_load_supermimic() {
  if (obs_input_name_.empty()) obs_input_name_ = "obs";
  if (action_output_name_.empty()) action_output_name_ = "actions";
  onnx_.require_inputs({obs_input_name_});
  onnx_.require_outputs({action_output_name_});
  const auto& shapes = onnx_.input_shapes();
  if (shapes.size() != 1 || shapes[0].size() != 2 ||
      (shapes[0][0] != 1 && shapes[0][0] != -1) || shapes[0][1] != num_obs_) {
    throw std::runtime_error(variant_ + " ONNX input must match the configured packed observation dimension");
  }
  if (variant_ == "allmimic") allmimic_controller_.reset({1.f, 0.f, 0.f, 0.f}, false);
  else supermimic_controller_.reset({1.f, 0.f, 0.f, 0.f}, false);
}

void GenericPolicyPlayer::schema_summary_supermimic() {
  schema_summary_common();
  std::cout << "  " << variant_ << "_contract packed_obs=" << num_obs_
            << " obs_frame=" << (variant_ == "allmimic" ? 93 : 99) << " history=10 goal=730 actions=29"
            << " reference=" << trajectory_path_.string()
            << " yaw_alignment=" << (supermimic_align_initial_yaw_ ? "initial" : "none")
            << " end_behavior=" << (supermimic_stop_at_end_ ? "stop" : "hold")
            << " preserve_pd_torque=" << command_safety_config_.preserve_pd_torque
            << " policy_step_source=" << (supermimic_use_lowstate_tick_ ? "lowstate_tick" : "wall_clock")
            << " policy_tick_ms=20 sim_lowstate_decimation=" << sim_lowstate_decimation_
            << " sim_lockstep_sequence=" << supermimic_sim_lockstep_sequence_ << "\n";
}

void GenericPolicyPlayer::check_only_supermimic() {
  if (variant_ == "allmimic") check_only_motion_controller(allmimic_controller_);
  else check_only_motion_controller(supermimic_controller_);
}

template <typename Controller>
void GenericPolicyPlayer::check_only_motion_controller(Controller& controller) {
  SuperMimicRobotFrame robot;
  std::copy(default_q_.begin(), default_q_.end(), robot.q.begin());
  const auto input = controller.build_input(robot);
  const auto action = onnx_.run_single(input, obs_input_name_, action_output_name_, 29);
  controller.decode_action(action);
  controller.reset(robot.quat, false);
  std::cout << "  " << variant_ << " obs_dim=" << input.size() << " actions=" << action.size()
            << " reference_frames=" << controller.reference_size()
            << " max_frames=" << max_frames_ << " inference_check=passed\n";
}

void GenericPolicyPlayer::control_loop_supermimic() {
  if (variant_ == "allmimic") control_loop_motion_controller(allmimic_controller_);
  else control_loop_motion_controller(supermimic_controller_);
}

template <typename Controller>
void GenericPolicyPlayer::control_loop_motion_controller(Controller& controller) {
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
  double inference_ms_last = 0.0;
  double inference_ms_total = 0.0;
  double inference_ms_max = 0.0;
  int64_t inference_count = 0;
  uint32_t last_policy_tick_ms = state.source_tick_ms;
  uint64_t last_policy_state_seq = state.seq;
  struct LatencySample {
    uint64_t state_seq;
    Clock::time_point received_at, started_at, input_ready_at, action_ready_at, target_ready_at;
  };
  std::mutex latency_mutex;
  std::vector<LatencySample> latency_samples;
  uint64_t target_sequence = 0;
  uint64_t last_recorded_sequence = 0;
  int64_t latency_count = 0;
  auto record_latency = [&](uint64_t sequence, Clock::time_point command_started_at,
                            Clock::time_point command_ready_at) {
    if (!args_.dry_run_latency || sequence == 0 || sequence <= last_recorded_sequence) return;
    LatencySample sample;
    {
      std::lock_guard<std::mutex> lock(latency_mutex);
      sample = latency_samples.at(sequence - 1);
    }
    const auto elapsed_ms = [](Clock::time_point begin, Clock::time_point end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    event_log_.log("dry_run_latency",
                   {event_string("variant", variant_), event_int("sequence", sequence),
                    event_int("state_seq", sample.state_seq), event_bool("lowcmd_sent", false),
                    event_int("command_crc", dry_run_command_crc_),
                    event_bool("synthetic_state", args_.offline_dry_run_state),
                    event_number("state_age_ms", elapsed_ms(sample.received_at, sample.started_at)),
                    event_number("preprocess_ms", elapsed_ms(sample.started_at, sample.input_ready_at)),
                    event_number("onnx_ms", elapsed_ms(sample.input_ready_at, sample.action_ready_at)),
                    event_number("postprocess_ms", elapsed_ms(sample.action_ready_at, sample.target_ready_at)),
                    event_number("publish_wait_ms", elapsed_ms(sample.target_ready_at, command_started_at)),
                    event_number("command_ms", elapsed_ms(command_started_at, command_ready_at)),
                    event_number("pipeline_ms", elapsed_ms(sample.started_at, command_ready_at)),
                    event_number("e2e_ms", elapsed_ms(sample.received_at, command_ready_at))});
    last_recorded_sequence = sequence;
    ++latency_count;
  };
  if (args_.dry_run_latency) {
    event_log_.log("dry_run_latency_started",
                   {event_string("variant", variant_), event_bool("lowcmd_sent", false),
                    event_int("onnx_threads", onnx_threads_), event_int("publish_hz", publish_hz)});
  }
  auto infer = [&](const RobotState& policy_state) {
    const auto pipeline_start = Clock::now();
    require_fresh(policy_state);
    const auto reference_frame = controller.reference_index();
    const auto input = controller.build_input(robot_frame(policy_state));
    const auto inference_start = Clock::now();
    const auto action = onnx_.run_single(input, obs_input_name_, action_output_name_, 29);
    const auto action_ready = Clock::now();
    const double inference_ms = std::chrono::duration<double, std::milli>(action_ready - inference_start).count();
    inference_ms_last = inference_ms;
    inference_ms_total += inference_ms;
    inference_ms_max = std::max(inference_ms_max, inference_ms);
    ++inference_count;
    auto next_target = controller.decode_action(action);
    controller.accept_action(action);
    target = std::move(next_target);
    target_kp = kp_;
    target_kd = kd_;
    policy_counter_ = static_cast<int64_t>(controller.policy_step());
    last_policy_tick_ms = policy_state.source_tick_ms;
    last_policy_state_seq = policy_state.seq;
    if (args_.dry_run_latency) {
      const auto target_ready = Clock::now();
      std::lock_guard<std::mutex> lock(latency_mutex);
      latency_samples.push_back({policy_state.seq, policy_state.stamp, pipeline_start,
                                 inference_start, action_ready, target_ready});
      target_sequence = latency_samples.size();
    }
    if (supermimic_trace_) {
      event_log_.log(variant_ == "allmimic" ? "allmimic_inference" : "supermimic_inference",
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
    controller.reset(state.quat, supermimic_align_initial_yaw_);
    const auto saved_controller = controller;
    infer(state);
    const auto decision = wait_preview_decision(target, target_kp, target_kd, state.mode_machine, publish_hz, 0);
    if (decision != PreviewDecision::kCommit) {
      controller = saved_controller;
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
    controller.reset(state.quat, supermimic_align_initial_yaw_);
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
    const auto command_started_at = Clock::now();
    publish(target, target_kp, target_kd, snapshot().mode_machine, false, CommandPhase::kActive);
    const auto now = Clock::now();
    record_latency(target_sequence, command_started_at, now);
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
        [&, this](const JointCommand& command, bool active) {
          const auto command_started_at = Clock::now();
          const auto current_state = snapshot();
          publish(command.q, command.kp, command.kd, current_state.mode_machine, false,
                  active ? CommandPhase::kActive : CommandPhase::kHold);
          const auto command_ready_at = Clock::now();
          if (active) record_latency(command.sequence, command_started_at, command_ready_at);
        }, preview_committed);
  }
  std::cout << player_log_prefix() << " start motion_conditioned_tracking/" << variant_
            << " hz=50 publish_hz=" << publish_hz
            << " reference_frames=" << controller.frame_limit()
            << " end_behavior=" << (supermimic_stop_at_end_ ? "stop" : "hold")
            << " preserve_pd_torque=" << command_safety_config_.preserve_pd_torque
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
        if (supermimic_stop_at_end_ && controller.at_end()) {
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
          async_publisher->update({target, target_kp, target_kd, target_sequence});
        }
        if (stats_interval_ > 0 && policy_counter_ % stats_interval_ == 0) {
          event_log_.log("runtime_metrics",
                         {event_string("player", player_label()), event_string("variant", variant_),
                          event_int("step", policy_counter_),
                          event_int("reference_frame", controller.reference_index()),
                          event_number("inference_ms_last", inference_ms_last),
                          event_number("inference_ms_max", inference_ms_max),
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
  if (args_.dry_run_latency) {
    event_log_.log("dry_run_latency_finished",
                   {event_int("inference_count", inference_count),
                    event_int("latency_count", latency_count),
                    event_int("unmeasured_count", inference_count - latency_count),
                    event_bool("lowcmd_sent", false)});
  }
  if (finished && !g_stop_requested.load() && !handoff_complete_file_.empty()) {
    if (!command_safety_config_.preserve_pd_torque &&
        last_published_q_.size() == static_cast<std::size_t>(motor_count_)) {
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
                  event_number("inference_ms_avg", inference_count > 0 ? inference_ms_total / inference_count : 0.0),
                  event_number("inference_ms_max", inference_ms_max),
                  event_int("policy_deadline_miss", policy_deadline_miss),
                  event_int("publish_deadline_miss", metrics.deadline_miss),
                  event_number("max_publish_interval_ms", metrics.max_publish_interval_ms),
                  event_string("phase", "final")});
}

}
