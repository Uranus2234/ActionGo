#include "action_go/runtime/pbfm_policy.h"
#include "action_go/runtime/policy_protocol.h"
#include "action_go/runtime/yaml_contract.h"
#include "action_go/runtime/depth_image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace action_go::runtime {
namespace {

std::filesystem::path relative_path(const std::filesystem::path& spec, const std::string& value) {
  const std::filesystem::path path(value);
  return (path.is_absolute() ? path : spec.parent_path() / path).lexically_normal();
}

void require_value(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("P-BFM " + message);
}

}

PbfmSpec PbfmSpec::load(const YAML::Node& root, const std::filesystem::path& spec_path) {
  namespace yamlc = yaml_contract;
  for (const auto& rule : protocol_yaml_dim_rules_for_variant("pbfm")) {
    require_value(yamlc::required_scalar<int>(root, rule.path) == rule.expected,
                  std::string(rule.path) + " does not match protocol");
  }
  require_value(yamlc::required_scalar<std::string>(root, "observation.history_timing") == "previous_only",
                "history_timing must be previous_only");
  require_value(yamlc::required_scalar<std::string>(root, "observation.history_reset") == "zero",
                "history_reset must be zero");
  require_value(yamlc::required_vector<std::string>(root, "observation.history_keys") ==
                  std::vector<std::string>{"actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"},
                "history_keys differ from training order");
  require_value(yamlc::required_scalar<float>(root, "observation.angular_velocity_scale") == 0.25f,
                "angular_velocity_scale must be 0.25");
  require_value(yamlc::required_scalar<std::string>(root, "sensor.profile") == "optical_z_maxcrop18_reset_v2",
                "unsupported depth profile");
  require_value(yamlc::required_scalar<std::string>(root, "sensor.quantity") == "optical_z",
                "depth quantity must be optical_z");
  require_value(yamlc::required_vector<int>(root, "sensor.source_shape") == std::vector<int>{36, 64} &&
                  yamlc::required_vector<int>(root, "sensor.crop") == std::vector<int>{18, 0, 16, 16} &&
                  yamlc::required_vector<int>(root, "sensor.sampled_ages") ==
                      std::vector<int>{35, 30, 25, 20, 15, 10, 5, 0},
                "depth source shape, crop or sampled_ages differ from training");
  require_value(yamlc::required_vector<int>(root, "observation.depth_input_shape") ==
                  std::vector<int>{1, 8, 18, 32}, "depth input shape must be [1,8,18,32]");
  require_value(yamlc::required_scalar<float>(root, "sensor.max_distance_m") == 2.5f &&
                  yamlc::required_scalar<float>(root, "sensor.invalid_fill_m") == 2.5f &&
                  yamlc::required_scalar<float>(root, "sensor.gaussian_sigma") == 1.f &&
                  yamlc::required_scalar<std::string>(root, "sensor.depth_reset") == "first_valid" &&
                  yamlc::required_scalar<int>(root, "sensor.fixed_delay_frames") == 0,
                "depth preprocessing or reset differs from public deployment contract");
  require_value(yamlc::required_scalar<int>(root, "runtime.inference_hz") == 50 &&
                  yamlc::required_scalar<int>(root, "runtime.warmup_steps") == 0,
                "requires 50Hz inference and zero history warmup steps");
  require_value(yamlc::required_scalar<std::string>(root, "action.type") == "target_q" &&
                  yamlc::required_scalar<bool>(root, "action.add_default") &&
                  yamlc::required_scalar<float>(root, "action.action_rescale") == 5.f &&
                  yamlc::required_vector<float>(root, "action.clip") == std::vector<float>{-1.f, 1.f},
                "action decoder must use default + scale * (5 * clip(raw,-1,1))");
  PbfmSpec spec;
  if (root["context"]["command_source"]) {
    spec.command_source = yamlc::required_scalar<std::string>(root, "context.command_source");
  }
  require_value(spec.command_source == "context" || spec.command_source == "wireless_remote",
                "context.command_source must be context or wireless_remote");
  if (spec.command_source == "wireless_remote") {
    spec.max_forward_speed = yamlc::required_scalar<float>(root, "context.max_forward_speed");
    spec.max_backward_speed = yamlc::required_scalar<float>(root, "context.max_backward_speed");
    spec.max_lateral_speed = yamlc::required_scalar<float>(root, "context.max_lateral_speed");
    spec.max_yaw_rate = yamlc::required_scalar<float>(root, "context.max_yaw_rate");
    for (float limit : {spec.max_forward_speed, spec.max_backward_speed,
                        spec.max_lateral_speed, spec.max_yaw_rate}) {
      require_value(std::isfinite(limit) && limit > 0.f, "remote command limits must be finite and positive");
    }
  }
  if (root["context"]["source"]) spec.context_source = yamlc::required_scalar<std::string>(root, "context.source");
  require_value(spec.context_source == "cache" || spec.context_source == "realtime_udp",
                "context.source must be cache or realtime_udp");
  if (spec.context_source == "realtime_udp") {
    require_value(spec.command_source == "context", "realtime context supplies its own heading and velocity");
    spec.context_udp_bind = yamlc::required_scalar<std::string>(root, "context.udp_bind");
    spec.context_udp_port = yamlc::required_scalar<int>(root, "context.udp_port");
    spec.context_timeout_sec = yamlc::required_scalar<double>(root, "context.timeout_sec");
    spec.context_startup_timeout_sec = yamlc::required_scalar<double>(root, "context.startup_timeout_sec");
    require_value(spec.context_udp_port > 0 && spec.context_udp_port <= 65535 &&
                    std::isfinite(spec.context_timeout_sec) && spec.context_timeout_sec > 0.0 &&
                    std::isfinite(spec.context_startup_timeout_sec) && spec.context_startup_timeout_sec > 0.0,
                  "invalid realtime context port or timeouts");
    spec.context.latent.assign(kBfmProtocolLatentDim, 0.f);
    spec.context.latent.front() = std::sqrt(static_cast<float>(kBfmProtocolLatentDim));
  } else {
    spec.context_path = relative_path(spec_path, yamlc::required_scalar<std::string>(root, "context.path"));
    const auto context_root = YAML::LoadFile(spec.context_path.string());
    if (yamlc::required_scalar<std::string>(context_root, "contract") == "tracking_heading0_yaw_velocity_v1") {
      spec.context_sequence = BfmContextSequence::load(spec.context_path, kBfmProtocolLatentDim);
      require_value(std::abs(spec.context_sequence.dt - 0.02) < 1e-9, "context sequence requires dt=0.02");
      spec.context = spec.context_sequence.frames.front();
    } else {
      spec.context = BfmContext::load(spec.context_path, kBfmProtocolLatentDim);
    }
  }
  if (root["context"]["idle_path"]) {
    require_value(spec.command_source == "wireless_remote", "idle_path requires wireless_remote commands");
    spec.idle_context_path = relative_path(spec_path, yamlc::required_scalar<std::string>(root, "context.idle_path"));
    spec.idle_context = BfmContext::load(spec.idle_context_path, kBfmProtocolLatentDim);
    require_value(spec.idle_context.velocity_command == std::array<float, 3>{0.f, 0.f, 1.f},
                  "idle context must command a valid zero velocity [0,0,1]");
    spec.command_deadband = yamlc::required_scalar<float>(root, "context.command_deadband");
    require_value(std::isfinite(spec.command_deadband) && spec.command_deadband >= 0.f &&
                    spec.command_deadband < 1.f, "command_deadband must be within [0,1)");
  }
  if (root["context"]["max_heading_lead_rad"]) {
    require_value(!spec.idle_context_path.empty(), "max_heading_lead_rad requires idle_path");
    spec.max_heading_lead = yamlc::required_scalar<float>(root, "context.max_heading_lead_rad");
    require_value(std::isfinite(spec.max_heading_lead) && spec.max_heading_lead > 0.f &&
                    spec.max_heading_lead < static_cast<float>(M_PI), "max_heading_lead_rad must be within (0,pi)");
  }
  if (root["context"]["max_translation_speed"]) {
    spec.max_translation_speed = yamlc::required_scalar<float>(root, "context.max_translation_speed");
    require_value(spec.command_source == "wireless_remote" && std::isfinite(spec.max_translation_speed) &&
                    spec.max_translation_speed > 0.f, "max_translation_speed requires remote commands and a positive limit");
  }
  if (root["context"]["proportional_speed"]) {
    spec.proportional_speed = yamlc::required_scalar<bool>(root, "context.proportional_speed");
    require_value(!spec.proportional_speed || root["context"]["directional_paths"],
                  "proportional_speed requires directional_paths");
  }
  if (root["context"]["directional_paths"]) {
    require_value(!spec.idle_context_path.empty(), "directional_paths requires idle_path and wireless_remote commands");
    for (const auto& name : {"forward", "backward", "left", "right"}) {
      const auto path = relative_path(spec_path, yamlc::required_scalar<std::string>(
          root, std::string("context.directional_paths.") + name));
      const std::string direction(name);
      BfmContext context;
      const auto directional_root = YAML::LoadFile(path.string());
      if (yamlc::required_scalar<std::string>(directional_root, "contract") == "tracking_heading0_yaw_velocity_v1") {
        auto sequence = BfmContextSequence::load(path, kBfmProtocolLatentDim);
        require_value(std::abs(sequence.dt - 0.02) < 1e-9, "directional sequence requires dt=0.02");
        spec.directional_sequences.emplace(direction, std::move(sequence));
        context = spec.directional_sequences.at(direction).frames.front();
      } else {
        context = BfmContext::load(path, kBfmProtocolLatentDim);
      }
      const auto validate_direction = [&](const BfmContext& frame) {
        const auto& velocity = frame.velocity_command;
        const bool forward = direction == "forward" || direction == "backward";
        const float speed = velocity[forward ? 0 : 1];
        require_value(velocity[2] == 1.f && velocity[forward ? 1 : 0] == 0.f &&
                        ((direction == "forward" || direction == "left") ? speed > 0.f : speed < 0.f),
                      "directional context velocity disagrees with " + direction);
      };
      validate_direction(context);
      if (spec.directional_sequences.count(direction)) {
        for (const auto& frame : spec.directional_sequences.at(direction).frames) validate_direction(frame);
      }
      spec.directional_contexts.emplace(name, std::move(context));
    }
  }
  if (root["context"]["turn_paths"]) {
    require_value(!spec.idle_context_path.empty(), "turn_paths requires idle_path");
    for (const auto& name : {"left", "right"}) {
      const auto path = relative_path(spec_path, yamlc::required_scalar<std::string>(
          root, std::string("context.turn_paths.") + name));
      const auto turn_root = YAML::LoadFile(path.string());
      const float yaw_rate = yamlc::required_scalar<float>(turn_root, "root_yaw_rate_rad_s");
      require_value(std::isfinite(yaw_rate) && (std::string(name) == "left" ? yaw_rate > 0.f : yaw_rate < 0.f),
                    "turn context yaw rate disagrees with direction");
      auto sequence = BfmContextSequence::load(path, kBfmProtocolLatentDim);
      require_value(std::abs(sequence.dt - 0.02) < 1e-9, "turn sequence requires dt=0.02");
      for (const auto& frame : sequence.frames) {
        require_value(frame.velocity_command == std::array<float, 3>{0.f, 0.f, 1.f},
                      "turn context must command zero translation");
      }
      spec.turn_sequences.emplace(name, std::move(sequence));
    }
  }
  spec.depth_source = yamlc::required_scalar<std::string>(root, "sensor.depth_source");
  require_value(spec.depth_source == "shm" || spec.depth_source == "file" || spec.depth_source == "realsense",
                "depth_source must be shm, file or realsense");
  spec.depth_timeout_sec = yamlc::required_scalar<double>(root, "sensor.timeout_sec");
  require_value(std::isfinite(spec.depth_timeout_sec) && spec.depth_timeout_sec > 0.0 &&
                  spec.depth_timeout_sec <= 0.5, "depth timeout must be within (0,0.5] seconds");
  if (spec.depth_source == "shm") {
    spec.shm_name = yamlc::required_scalar<std::string>(root, "sensor.shm_name");
    require_value(!spec.shm_name.empty(), "sensor.shm_name must not be empty");
  } else if (spec.depth_source == "realsense") {
    auto& camera = spec.realsense;
    camera.width = yamlc::required_scalar<int>(root, "sensor.realsense.width");
    camera.height = yamlc::required_scalar<int>(root, "sensor.realsense.height");
    camera.fps = yamlc::required_scalar<int>(root, "sensor.realsense.fps");
    camera.min_frames = yamlc::required_scalar<int>(root, "sensor.realsense.min_frames");
    camera.min_valid_fraction = yamlc::required_scalar<float>(root, "sensor.realsense.min_valid_fraction");
    camera.startup_timeout_sec = yamlc::required_scalar<double>(root, "sensor.realsense.startup_timeout_sec");
    camera.horizontal_fov_deg = yamlc::required_scalar<float>(root, "sensor.horizontal_fov_deg");
    camera.vertical_fov_deg = yamlc::required_scalar<float>(root, "sensor.vertical_fov_deg");
    camera.output_width = 64;
    camera.output_height = 36;
    camera.crop_top = 18;
    camera.crop_left = 16;
    camera.crop_right = 16;
    camera.invalid_fill_m = 2.5f;
    camera.timeout_sec = spec.depth_timeout_sec;
    require_value(camera.width > 0 && camera.height > 0 && camera.fps > 0 && camera.min_frames > 0 &&
                    std::isfinite(camera.startup_timeout_sec) && camera.startup_timeout_sec > 0.0,
                  "RealSense capture dimensions, rate and startup limits must be positive");
    require_value(std::isfinite(camera.min_valid_fraction) && camera.min_valid_fraction > 0.f &&
                    camera.min_valid_fraction <= 1.f,
                  "RealSense min_valid_fraction must be within (0,1]");
    require_value(std::abs(camera.horizontal_fov_deg - 89.51f) < 0.01f &&
                    std::abs(camera.vertical_fov_deg - 58.29f) < 0.01f,
                  "RealSense target field of view must match training (89.51 x 58.29 degrees)");
    const auto mount = yamlc::required_vector_size<double>(root, "sensor.mount_position_torso_m", 3);
    const auto optical = yamlc::required_vector_size<double>(root, "sensor.optical_quat_torso_xyzw", 4);
    const std::array<double, 3> expected_mount{0.0487988662332928, 0.01, 0.4378029937970051};
    const std::array<double, 4> expected_optical{-0.6579550475696607, 0.6623183568544073,
                                               -0.25121840449386046, 0.25558171377860706};
    for (size_t index = 0; index < mount.size(); ++index)
      require_value(std::abs(mount[index] - expected_mount[index]) < 1e-6, "camera mount must match training");
    for (size_t index = 0; index < optical.size(); ++index)
      require_value(std::abs(optical[index] - expected_optical[index]) < 1e-6, "optical quaternion must match training");
  } else {
    spec.depth_file = relative_path(spec_path, yamlc::required_scalar<std::string>(root, "sensor.path"));
    require_value(std::filesystem::exists(spec.depth_file), "depth file not found: " + spec.depth_file.string());
  }
  return spec;
}

std::vector<float> PbfmDepthHistory::append(const std::vector<float>& optical_z_m) {
  auto frame = preprocess_pbfm_depth(optical_z_m);
  if (frames_.empty()) frames_.assign(37, frame);
  else {
    frames_.pop_front();
    frames_.push_back(std::move(frame));
  }
  std::vector<float> output;
  output.reserve(8 * 18 * 32);
  for (size_t age : {35, 30, 25, 20, 15, 10, 5, 0}) {
    const auto& sampled = frames_.at(36 - age);
    output.insert(output.end(), sampled.begin(), sampled.end());
  }
  return output;
}

BfmContext PbfmSpec::movement_context(float forward_speed, float lateral_speed, size_t reference_step) const {
  if (forward_speed == 0.f && lateral_speed == 0.f) return idle_context;
  const auto directional_context = [&](const std::string& direction) {
    const auto sequence = directional_sequences.find(direction);
    if (sequence == directional_sequences.end()) return directional_contexts.at(direction);
    return sequence->second.frames.at(reference_step % sequence->second.frames.size());
  };
  const auto apply_strength = [&](BfmContext context) {
    if (!proportional_speed) return context;
    const float forward_limit = forward_speed >= 0.f ? max_forward_speed : max_backward_speed;
    const float strength = std::max(std::abs(forward_speed) / forward_limit,
                                    std::abs(lateral_speed) / max_lateral_speed);
    return blend_with_idle(std::move(context), std::min(1.f, strength));
  };
  if (lateral_speed == 0.f) return apply_strength(directional_context(forward_speed > 0.f ? "forward" : "backward"));
  if (forward_speed == 0.f) return apply_strength(directional_context(lateral_speed > 0.f ? "left" : "right"));
  auto context = directional_context(forward_speed > 0.f ? "forward" : "backward");
  const auto lateral = directional_context(lateral_speed > 0.f ? "left" : "right");
  const float forward_weight = std::abs(forward_speed) / std::abs(context.velocity_command[0]);
  const float lateral_weight = std::abs(lateral_speed) / std::abs(lateral.velocity_command[1]);
  float norm_squared = 0.f;
  float reference_norm_squared = 0.f;
  for (size_t index = 0; index < context.latent.size(); ++index) {
    reference_norm_squared += context.latent[index] * context.latent[index];
    context.latent[index] = forward_weight * context.latent[index] + lateral_weight * lateral.latent[index];
    norm_squared += context.latent[index] * context.latent[index];
  }
  require_value(norm_squared > 1e-8f, "directional latent blend is degenerate");
  const float scale = std::sqrt(reference_norm_squared / norm_squared);
  for (auto& value : context.latent) value *= scale;
  return apply_strength(std::move(context));
}

BfmContext PbfmSpec::blend_with_idle(BfmContext context, float strength) const {
  const float amount = std::clamp(strength, 0.f, 1.f);
  if (amount == 0.f) return idle_context;
  if (amount == 1.f) return context;
  float norm_squared = 0.f;
  float reference_norm_squared = 0.f;
  for (size_t index = 0; index < context.latent.size(); ++index) {
    reference_norm_squared += context.latent[index] * context.latent[index];
    context.latent[index] = amount * context.latent[index] + (1.f - amount) * idle_context.latent[index];
    norm_squared += context.latent[index] * context.latent[index];
  }
  require_value(norm_squared > 1e-8f, "idle/motion latent blend is degenerate");
  const float scale = std::sqrt(reference_norm_squared / norm_squared);
  for (auto& value : context.latent) value *= scale;
  return context;
}

BfmContext PbfmSpec::turning_context(float yaw_rate, size_t reference_step) const {
  if (yaw_rate == 0.f || turn_sequences.empty()) return idle_context;
  const auto& sequence = turn_sequences.at(yaw_rate > 0.f ? "left" : "right");
  auto context = sequence.frames.at(reference_step % sequence.frames.size());
  const float strength = std::min(1.f, std::abs(yaw_rate) / max_yaw_rate);
  if (strength == 1.f) return context;
  const auto& opposite = turn_sequences.at(yaw_rate > 0.f ? "right" : "left");
  const auto& other = opposite.frames.at(reference_step % opposite.frames.size());
  const float weight = (1.f + strength) / 2.f;
  float norm_squared = 0.f;
  float reference_norm_squared = 0.f;
  for (size_t index = 0; index < context.latent.size(); ++index) {
    reference_norm_squared += context.latent[index] * context.latent[index];
    context.latent[index] = weight * context.latent[index] + (1.f - weight) * other.latent[index];
    norm_squared += context.latent[index] * context.latent[index];
  }
  require_value(norm_squared > 1e-8f, "turn latent blend is degenerate");
  const float scale = std::sqrt(reference_norm_squared / norm_squared);
  for (auto& value : context.latent) value *= scale;
  return context;
}

void PbfmHeadingController::reset(float yaw) {
  target_yaw_ = yaw;
  was_turning_ = false;
}

float PbfmHeadingController::update(float axis, float max_rate, float max_lead, float current_yaw, float dt) {
  const float full_turn = 2.f * static_cast<float>(M_PI);
  target_yaw_ = std::remainder(target_yaw_ + axis * max_rate * dt, full_turn);
  if (max_lead > 0.f) {
    if (axis != 0.f) {
      const float error = std::remainder(target_yaw_ - current_yaw, full_turn);
      target_yaw_ = std::remainder(current_yaw + std::clamp(error, -max_lead, max_lead), full_turn);
    } else if (was_turning_) {
      target_yaw_ = current_yaw;
    }
  }
  was_turning_ = axis != 0.f;
  return target_yaw_;
}

void PbfmPolicy::validate_model(const OnnxRunner& runner) {
  const std::vector<std::string> names{
      "state", "last_action", "history_actor", "heading", "velocity_command", "depth_image", "z"};
  const std::vector<std::vector<int64_t>> shapes{
      {1, kBfmProtocolStateDim}, {1, kUnitreeG1ProtocolMotorCount}, {1, kBfmProtocolHistoryDim},
      {1, kPbfmProtocolHeadingDim}, {1, kPbfmProtocolVelocityDim},
      {1, kPbfmProtocolDepthFrames, kPbfmProtocolDepthHeight, kPbfmProtocolDepthWidth},
      {1, kBfmProtocolLatentDim}};
  require_value(runner.input_names() == names && runner.output_names() ==
                  std::vector<std::string>{"action", "estimated_velocity"}, "ONNX I/O signature mismatch");
  for (size_t input = 0; input < names.size(); ++input) {
    auto actual = runner.input_shapes().at(input);
    if (!actual.empty() && actual[0] < 0) actual[0] = 1;
    require_value(actual == shapes[input], "ONNX shape mismatch for " + names[input]);
  }
}

std::unordered_map<std::string, OnnxRunner::TensorInput> PbfmPolicy::inputs(
    const BfmObservation& observation, const BfmContext& context,
    const std::array<float, 2>& heading, const std::vector<float>& depth) {
  require_value(depth.size() == kPbfmProtocolDepthFrames * kPbfmProtocolDepthHeight * kPbfmProtocolDepthWidth,
                "depth stack size mismatch");
  for (float value : depth) {
    require_value(std::isfinite(value) && value >= 0.f && value <= 1.f,
                  "depth must be normalized float32 in [0,1]");
  }
  return {{"state", {observation.state, {1, kBfmProtocolStateDim}}},
          {"last_action", {observation.last_action, {1, kUnitreeG1ProtocolMotorCount}}},
          {"history_actor", {observation.history, {1, kBfmProtocolHistoryDim}}},
          {"heading", {std::vector<float>(heading.begin(), heading.end()), {1, kPbfmProtocolHeadingDim}}},
          {"velocity_command", {std::vector<float>(context.velocity_command.begin(), context.velocity_command.end()),
                                  {1, kPbfmProtocolVelocityDim}}},
          {"depth_image", {depth, {1, kPbfmProtocolDepthFrames, kPbfmProtocolDepthHeight, kPbfmProtocolDepthWidth}}},
          {"z", {context.latent, {1, kBfmProtocolLatentDim}}}};
}

PbfmResult PbfmPolicy::infer(OnnxRunner& runner, const BfmObservation& observation,
                           const BfmContext& context, const std::array<float, 2>& heading,
                           const std::vector<float>& depth) {
  const auto output = runner.run_named_typed(inputs(observation, context, heading, depth),
      {{"action", {OnnxRunner::TensorElementType::kFloat32, {1, kUnitreeG1ProtocolMotorCount}}},
       {"estimated_velocity", {OnnxRunner::TensorElementType::kFloat32, {1, kPbfmProtocolVelocityDim}}}});
  return {std::get<std::vector<float>>(output.at("action").values),
          std::get<std::vector<float>>(output.at("estimated_velocity").values)};
}

}
