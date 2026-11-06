#include "generic_policy_runtime_internal.h"
#include "action_go/runtime/input_loss.h"

#include <nlohmann/json.hpp>

namespace action_go::runtime::generic_policy_runtime {

void GenericPolicyPlayer::load_spec_pbfm(const YAML::Node& root) {
    load_spec_bfm(root);
    YAML::Node configured = YAML::Clone(root);
    if (!args_.depth_source_override.empty()) configured["sensor"]["depth_source"] = args_.depth_source_override;
    if (!args_.depth_shm_override.empty()) configured["sensor"]["shm_name"] = args_.depth_shm_override;
    pbfm_spec_ = PbfmSpec::load(configured, spec_path_);
    if (inference_hz_ != 50) throw std::runtime_error("P-BFM requires 50Hz inference");
    if (!args_.latent_file_override.empty()) {
      throw std::runtime_error("P-BFM uses a complete context; --latent-file cannot override it");
    }
    if (pbfm_spec_.depth_source == "file" && !args_.check_only && !args_.dry_run) {
      throw std::runtime_error("P-BFM file depth is only allowed for --check-only or --dry-run");
    }
  }

void GenericPolicyPlayer::post_load_pbfm() {
    PbfmPolicy::validate_model(onnx_);
    if (pbfm_spec_.depth_source == "file") {
      const auto bytes = read_binary_file(pbfm_spec_.depth_file);
      if (npy::parse_header(bytes).shape != std::vector<size_t>{8, 18, 32}) {
        throw std::runtime_error("P-BFM offline depth must have shape [8,18,32]");
      }
      pbfm_file_depth_ = npy_to_float_vector(bytes, pbfm_spec_.depth_file.string());
    }
    BfmObservationHistory history;
    history.reset(motor_count_, bfm_history_steps_);
    const std::vector<float> zero(motor_count_, 0.f);
    const auto observation = history.build(default_q_, zero, default_q_, {0.f, 0.f, -1.f},
                                            {0.f, 0.f, 0.f}, zero, BfmHistoryTiming::kPreviousOnly);
    const auto depth = pbfm_file_depth_.empty() ? std::vector<float>(
        kPbfmProtocolDepthFrames * kPbfmProtocolDepthHeight * kPbfmProtocolDepthWidth, 1.f) : pbfm_file_depth_;
    PbfmPolicy::infer(onnx_, observation, pbfm_spec_.context,
                     pbfm_spec_.context.heading_input(0.f, 0.f), depth);
    if (!pbfm_spec_.idle_context_path.empty()) {
      PbfmPolicy::infer(onnx_, observation, pbfm_spec_.idle_context,
                       pbfm_spec_.idle_context.heading_input(0.f, 0.f), depth);
    }
    for (const auto& entry : pbfm_spec_.directional_contexts) {
      PbfmPolicy::infer(onnx_, observation, entry.second, entry.second.heading_input(0.f, 0.f), depth);
    }
    for (const auto& entry : pbfm_spec_.turn_sequences) {
      const auto& context = entry.second.frames.front();
      PbfmPolicy::infer(onnx_, observation, context, context.heading_input(0.f, 0.f), depth);
    }
    if (!args_.trace_file.empty()) {
      pbfm_trace_.open(args_.trace_file);
      if (!pbfm_trace_) throw std::runtime_error("cannot open P-BFM trace: " + args_.trace_file);
      pbfm_trace_ << std::setprecision(9);
    }
  }

void GenericPolicyPlayer::check_only_pbfm() {
    std::cout << "  pbfm state=64 history=372 z=256 depth=[1,8,18,32] actions=29 estimated_velocity=3\n"
              << "  context=" << pbfm_spec_.context_path.string()
              << " history_timing=previous_only depth_source=" << pbfm_spec_.depth_source << "\n"
              << "  context_frames=" << pbfm_spec_.context_sequence.frames.size()
              << " command_source=" << pbfm_spec_.command_source << "\n"
              << "  context_source=" << pbfm_spec_.context_source
              << " udp_port=" << pbfm_spec_.context_udp_port << "\n"
              << "  idle_context=" << pbfm_spec_.idle_context_path.string() << "\n"
              << "  directional_contexts=" << pbfm_spec_.directional_contexts.size() << "\n"
              << "  directional_sequences=" << pbfm_spec_.directional_sequences.size() << "\n"
              << "  turn_sequences=" << pbfm_spec_.turn_sequences.size()
              << " proportional_speed=" << pbfm_spec_.proportional_speed
              << " max_heading_lead_rad=" << pbfm_spec_.max_heading_lead << "\n"
              << "  synthetic ONNX check only; live depth readiness is checked before ready_file\n";
  }

std::vector<float> GenericPolicyPlayer::pbfm_depth_observation() {
    if (pbfm_spec_.depth_source == "file") return pbfm_file_depth_;
    std::vector<float> depth;
    try {
    if (pbfm_spec_.depth_source == "realsense") {
      if (!pbfm_realsense_) {
        pbfm_realsense_ = std::make_unique<RealSenseDepthStream>(pbfm_spec_.realsense);
      }
      depth = pbfm_depth_history_.append(pbfm_realsense_->observation());
    } else {
#ifdef ACTION_GO_HAS_ZMQ
      if (!pbfm_depth_stream_) {
        pbfm_depth_stream_ = std::make_unique<SharedMemoryDepthStream>(pbfm_spec_.shm_name,
            kPbfmProtocolDepthFrames * kPbfmProtocolDepthHeight * kPbfmProtocolDepthWidth,
            pbfm_spec_.depth_timeout_sec);
      }
      depth = pbfm_depth_stream_->observation();
      if (pbfm_depth_stream_->age_sec() > pbfm_spec_.depth_timeout_sec) {
        throw std::runtime_error("depth stream is stale");
      }
      for (float value : depth) {
        if (value < 0.f || value > 1.f) throw std::runtime_error("depth is outside [0,1]");
      }
#else
      throw std::runtime_error("P-BFM live depth requires a build with libzmq");
#endif
    }
    pbfm_depth_failures_ = 0;
    pbfm_last_depth_ = depth;
    return depth;
    } catch (const action_go::runtime::InputLoss&) {
      throw;
    } catch (const std::exception& error) {
      if (++pbfm_depth_failures_ >= 2) {
        throw action_go::runtime::InputLoss(std::string("P-BFM depth stream lost: ") + error.what());
      }
      if (pbfm_last_depth_.empty()) throw action_go::runtime::InputLoss(error.what());
      return pbfm_last_depth_;
    }
  }

std::vector<float> GenericPolicyPlayer::infer_pbfm(const RobotState& state, const BfmObservation& observation) {
    const auto depth = pbfm_depth_observation();
    const float current_yaw = quat_yaw(state.quat);
    const bool hold_to_move = !pbfm_spec_.idle_context_path.empty();
    const auto reference_step = hold_to_move ? pbfm_reference_step_ : static_cast<size_t>(policy_counter_);
    auto context = pbfm_spec_.context_sequence.frames.empty() ? pbfm_spec_.context :
        pbfm_spec_.context_sequence.at(hold_to_move ? reference_step % pbfm_spec_.context_sequence.frames.size() :
                                     reference_step, pbfm_initial_yaw_, current_yaw);
    std::string motion_state = "context";
    if (pbfm_context_stream_) {
      try {
        context = pbfm_context_stream_->current();
        pbfm_context_failures_ = 0;
        pbfm_last_context_ = context;
      } catch (const std::exception& error) {
        if (++pbfm_context_failures_ >= 2) {
          throw action_go::runtime::InputLoss(std::string("P-BFM realtime context lost: ") + error.what());
        }
        context = pbfm_last_context_;
      }
      context = context.aligned(pbfm_reference_start_yaw_, pbfm_initial_yaw_, current_yaw);
      motion_state = "pico_realtime";
    }
    std::array<float, 3> raw_command_axes{0.f, 0.f, 0.f};
    std::array<float, 3> command_axes{0.f, 0.f, 0.f};
    if (pbfm_spec_.command_source == "wireless_remote") {
      raw_command_axes = {sanitize_axis(unpack_float_le(state.wireless_remote, 20)),
                          -sanitize_axis(unpack_float_le(state.wireless_remote, 4)),
                          -sanitize_axis(unpack_float_le(state.wireless_remote, 8))};
      const auto command_axis = [&](size_t offset) {
        const float axis = sanitize_axis(unpack_float_le(state.wireless_remote, offset));
        return hold_to_move && std::abs(axis) <= pbfm_spec_.command_deadband ? 0.f : axis;
      };
      const float forward_axis = command_axis(20);
      const float lateral_axis = -command_axis(4);
      const float yaw_axis = -command_axis(8);
      command_axes = {forward_axis, lateral_axis, yaw_axis};
      float vx = forward_axis * (forward_axis >= 0.f ? pbfm_spec_.max_forward_speed :
                                                      pbfm_spec_.max_backward_speed);
      float vy = lateral_axis * pbfm_spec_.max_lateral_speed;
      const float speed = std::hypot(vx, vy);
      if (pbfm_spec_.max_translation_speed > 0.f && speed > pbfm_spec_.max_translation_speed) {
        const float scale = pbfm_spec_.max_translation_speed / speed;
        vx *= scale;
        vy *= scale;
      }
      const bool moving = forward_axis != 0.f || lateral_axis != 0.f || yaw_axis != 0.f;
      if (hold_to_move) {
        motion_state = moving ? "walking" : "idle";
        if (!pbfm_spec_.directional_contexts.empty()) {
          context = pbfm_spec_.movement_context(vx, vy, reference_step);
          if (vx == 0.f && vy == 0.f && yaw_axis != 0.f) {
            motion_state = "turning";
            context = pbfm_spec_.turning_context(yaw_axis * pbfm_spec_.max_yaw_rate, reference_step);
          }
          if (vx != 0.f || vy != 0.f || (yaw_axis != 0.f && !pbfm_spec_.turn_sequences.empty())) {
            ++pbfm_reference_step_;
          }
        } else if (!moving) {
          context = pbfm_spec_.idle_context;
        } else if (!pbfm_spec_.context_sequence.frames.empty()) {
          pbfm_reference_step_ = (reference_step + 1) % pbfm_spec_.context_sequence.frames.size();
        }
      } else {
        motion_state = "remote";
      }
      pbfm_command_yaw_ = pbfm_heading_controller_.update(yaw_axis, pbfm_spec_.max_yaw_rate,
          pbfm_spec_.max_heading_lead, current_yaw, 1.f / static_cast<float>(inference_hz_));
      const float relative = pbfm_command_yaw_ - pbfm_initial_yaw_;
      context.heading_relative = {std::cos(relative), std::sin(relative)};
      context.heading_valid = true;
      const float error = pbfm_command_yaw_ - current_yaw;
      context.velocity_command = {std::cos(error) * vx - std::sin(error) * vy,
                                  std::sin(error) * vx + std::cos(error) * vy, 1.f};
    }
    const auto heading = context.heading_input(pbfm_initial_yaw_, current_yaw);
    auto result = PbfmPolicy::infer(onnx_, observation, context, heading, depth);
    pbfm_estimated_velocity_ = result.estimated_velocity;
    if (event_log_.enabled() && (motion_state != pbfm_last_motion_state_ ||
                               policy_counter_ % inference_hz_ == 0)) {
      const auto latest = depth.end() - kPbfmProtocolDepthHeight * kPbfmProtocolDepthWidth;
      float depth_sum = 0.f;
      size_t far_pixels = 0;
      for (auto pixel = latest; pixel != depth.end(); ++pixel) {
        depth_sum += *pixel;
        far_pixels += *pixel >= 1.f;
      }
      const auto pixels = static_cast<float>(depth.end() - latest);
      event_log_.log("pbfm_command_state",
                     {event_int("step", policy_counter_),
                      event_string("motion_state", motion_state),
                      event_int("context_sequence", pbfm_context_stream_ ? pbfm_context_stream_->sequence() : -1),
                      event_int("reference_step", reference_step),
                      event_number("raw_forward_axis", raw_command_axes[0]),
                      event_number("raw_lateral_axis", raw_command_axes[1]),
                      event_number("raw_yaw_axis", raw_command_axes[2]),
                      event_number("forward_axis", command_axes[0]),
                      event_number("lateral_axis", command_axes[1]),
                      event_number("yaw_axis", command_axes[2]),
                      event_number("cmd_vx", context.velocity_command[0]),
                      event_number("cmd_vy", context.velocity_command[1]),
                      event_number("command_yaw", pbfm_command_yaw_),
                      event_number("current_yaw", current_yaw),
                      event_number("estimated_vx", result.estimated_velocity[0]),
                      event_number("estimated_vy", result.estimated_velocity[1]),
                      event_number("estimated_vz", result.estimated_velocity[2]),
                      event_number("depth_mean_m", depth_sum * 2.5f / pixels),
                      event_string("depth_stream", pbfm_realsense_ ? pbfm_realsense_->stats() :
                                   "source=" + pbfm_spec_.depth_source),
                      event_number("depth_far_fraction", far_pixels / pixels)});
    }
    pbfm_last_motion_state_ = motion_state;
    if (pbfm_trace_) {
      const auto decoded = decode_bfm_action(result.action, bfm_action_rescale_, default_q_, action_scale_,
                                             lower_limits_, upper_limits_);
      pbfm_trace_ << nlohmann::json{
          {"step", policy_counter_}, {"state", observation.state}, {"last_action", observation.last_action},
          {"initial_yaw", pbfm_initial_yaw_}, {"current_yaw", current_yaw},
          {"motion_state", motion_state}, {"reference_step", reference_step},
          {"context_sequence", pbfm_context_stream_ ? pbfm_context_stream_->sequence() : -1},
          {"command_axes", command_axes}, {"command_yaw", pbfm_command_yaw_},
          {"history_actor", observation.history}, {"heading", heading},
          {"velocity_command", context.velocity_command}, {"depth_image", depth},
          {"z", context.latent}, {"action", result.action},
          {"estimated_velocity", result.estimated_velocity}, {"scaled_action", decoded.scaled_action},
          {"target_q_before_safety", decoded.target_position}, {"kp", kp_}, {"kd", kd_}}.dump() << '\n';
    }
    return result.action;
  }

}
