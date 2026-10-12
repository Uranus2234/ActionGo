#include "action_go/runtime/sonic_planner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace action_go::runtime {
namespace {

using Quat = std::array<float, 4>;

struct PacketField {
  std::string dtype;
  std::vector<size_t> shape;
  const uint8_t* data{nullptr};
  size_t elements{0};
};

struct ParsedPacket {
  std::unordered_map<std::string, PacketField> fields;
  std::vector<std::string> order;
  std::string signature;
};

size_t dtype_size(const std::string& dtype) {
  if (dtype == "u8" || dtype == "bool") return 1;
  if (dtype == "i32" || dtype == "f32") return 4;
  if (dtype == "i64" || dtype == "f64") return 8;
  throw std::runtime_error("SONIC ZMQ unsupported dtype: " + dtype);
}

ParsedPacket parse_packet_header(const std::vector<uint8_t>& packet,
                                 const std::string& topic) {
  if (packet.size() < topic.size() ||
      std::memcmp(packet.data(), topic.data(), topic.size()) != 0) {
    throw std::runtime_error("SONIC ZMQ topic prefix mismatch");
  }
  if (packet.size() < topic.size() + kSonicZmqHeaderSize) {
    throw std::runtime_error("SONIC ZMQ packet is shorter than the 1280-byte header");
  }
  const char* raw = reinterpret_cast<const char*>(packet.data() + topic.size());
  const size_t length = strnlen(raw, kSonicZmqHeaderSize);
  const auto header = nlohmann::json::parse(std::string(raw, length));
  if (header.value("v", 0) != 1) {
    throw std::runtime_error("SONIC planner/command protocol version must be 1");
  }
  if (header.value("endian", std::string("")) != "le") {
    throw std::runtime_error("SONIC planner/command protocol requires little-endian payloads");
  }
  if (header.value("count", 0) != 1 || !header.contains("fields") ||
      !header.at("fields").is_array()) {
    throw std::runtime_error("SONIC planner/command header is invalid");
  }

  ParsedPacket parsed;
  std::ostringstream signature;
  signature << "v1:le";
  size_t offset = topic.size() + kSonicZmqHeaderSize;
  for (const auto& item : header.at("fields")) {
    const std::string name = item.at("name").get<std::string>();
    PacketField field;
    field.dtype = item.at("dtype").get<std::string>();
    field.shape = item.at("shape").get<std::vector<size_t>>();
    field.elements = 1;
    for (size_t dim : field.shape) {
      if (dim != 0 && field.elements > std::numeric_limits<size_t>::max() / dim) {
        throw std::runtime_error("SONIC ZMQ field shape overflows: " + name);
      }
      field.elements *= dim;
    }
    const size_t width = dtype_size(field.dtype);
    if (field.elements > std::numeric_limits<size_t>::max() / width) {
      throw std::runtime_error("SONIC ZMQ field byte size overflows: " + name);
    }
    const size_t bytes = field.elements * width;
    if (offset > packet.size() || bytes > packet.size() - offset) {
      throw std::runtime_error("SONIC ZMQ field exceeds packet bounds: " + name);
    }
    field.data = packet.data() + offset;
    offset += bytes;
    signature << ';' << name << ':' << field.dtype << ':';
    for (size_t dim : field.shape) signature << dim << ',';
    parsed.order.push_back(name);
    if (!parsed.fields.emplace(name, std::move(field)).second) {
      throw std::runtime_error("SONIC ZMQ duplicate field: " + name);
    }
  }
  if (offset != packet.size()) {
    throw std::runtime_error("SONIC ZMQ packet contains trailing payload bytes");
  }
  parsed.signature = signature.str();
  return parsed;
}

const PacketField& require_field(const ParsedPacket& packet, const std::string& name,
                                 const std::string& dtype,
                                 const std::vector<size_t>& shape) {
  const auto it = packet.fields.find(name);
  if (it == packet.fields.end()) throw std::runtime_error("SONIC ZMQ missing field: " + name);
  if (it->second.dtype != dtype || it->second.shape != shape) {
    throw std::runtime_error("SONIC ZMQ field contract mismatch: " + name);
  }
  return it->second;
}

template <typename T>
T read_scalar(const PacketField& field, size_t index = 0) {
  T value{};
  std::memcpy(&value, field.data + index * sizeof(T), sizeof(T));
  return value;
}

void require_finite(float value, const char* name) {
  if (!std::isfinite(value)) throw std::runtime_error(std::string("SONIC planner ") + name + " is NaN/Inf");
}

Quat normalize_quat(Quat q) {
  float norm2 = 0.f;
  for (float value : q) norm2 += value * value;
  if (!std::isfinite(norm2) || norm2 < 1e-12f) {
    throw std::runtime_error("SONIC planner quaternion is invalid");
  }
  const float inv = 1.f / std::sqrt(norm2);
  for (float& value : q) value *= inv;
  return q;
}

Quat slerp(Quat a, Quat b, float t) {
  a = normalize_quat(a);
  b = normalize_quat(b);
  float dot = 0.f;
  for (int i = 0; i < 4; ++i) dot += a[i] * b[i];
  if (dot < 0.f) {
    dot = -dot;
    for (float& value : b) value = -value;
  }
  dot = std::clamp(dot, -1.f, 1.f);
  if (dot > 0.9995f) {
    Quat result{};
    for (int i = 0; i < 4; ++i) result[i] = a[i] + t * (b[i] - a[i]);
    return normalize_quat(result);
  }
  const float theta = std::acos(dot);
  const float sin_theta = std::sin(theta);
  const float wa = std::sin((1.f - t) * theta) / sin_theta;
  const float wb = std::sin(t * theta) / sin_theta;
  Quat result{};
  for (int i = 0; i < 4; ++i) result[i] = wa * a[i] + wb * b[i];
  return normalize_quat(result);
}

bool near(float a, float b, float epsilon = 1e-5f) {
  return std::abs(a - b) <= epsilon;
}

bool same_vector(const std::array<float, 3>& a, const std::array<float, 3>& b) {
  return near(a[0], b[0]) && near(a[1], b[1]) && near(a[2], b[2]);
}

SonicPlannerMotionFrame sample_motion(const std::vector<SonicPlannerMotionFrame>& motion,
                                      float frame) {
  if (motion.empty()) throw std::runtime_error("SONIC planner motion is empty");
  frame = std::clamp(frame, 0.f, static_cast<float>(motion.size() - 1));
  const size_t f0 = static_cast<size_t>(std::floor(frame));
  const size_t f1 = std::min(f0 + 1, motion.size() - 1);
  const float w1 = frame - static_cast<float>(f0);
  const float w0 = 1.f - w1;
  SonicPlannerMotionFrame result;
  for (int i = 0; i < 3; ++i) {
    result.root_position[i] = w0 * motion[f0].root_position[i] + w1 * motion[f1].root_position[i];
  }
  result.body_quat_wxyz = slerp(motion[f0].body_quat_wxyz, motion[f1].body_quat_wxyz, w1);
  for (int i = 0; i < kSonicMotorCount; ++i) {
    result.joint_pos[i] = w0 * motion[f0].joint_pos[i] + w1 * motion[f1].joint_pos[i];
    result.joint_vel[i] = w0 * motion[f0].joint_vel[i] + w1 * motion[f1].joint_vel[i];
  }
  return result;
}

}  // namespace

SonicPlannerCommand parse_sonic_planner_packet(
    const std::vector<uint8_t>& packet, const std::string& topic,
    std::string* layout_signature) {
  const ParsedPacket parsed = parse_packet_header(packet, topic);
  static const std::vector<std::string> expected{"mode", "movement", "facing", "speed", "height"};
  if (parsed.order != expected) {
    throw std::runtime_error(
        "SONIC planner extensions (frozen upper body, VR-3PT, or Dex3) are not supported");
  }
  SonicPlannerCommand command;
  command.mode = read_scalar<int32_t>(require_field(parsed, "mode", "i32", {1}));
  const auto& movement = require_field(parsed, "movement", "f32", {3});
  const auto& facing = require_field(parsed, "facing", "f32", {3});
  for (int i = 0; i < 3; ++i) {
    command.movement[i] = read_scalar<float>(movement, i);
    command.facing[i] = read_scalar<float>(facing, i);
    require_finite(command.movement[i], "movement");
    require_finite(command.facing[i], "facing");
  }
  command.speed = read_scalar<float>(require_field(parsed, "speed", "f32", {1}));
  command.height = read_scalar<float>(require_field(parsed, "height", "f32", {1}));
  require_finite(command.speed, "speed");
  require_finite(command.height, "height");
  if (command.mode < 0 || command.mode >= kSonicPlannerModeCountV2) {
    throw std::runtime_error("SONIC planner mode must be in [0,26]");
  }
  if (layout_signature) *layout_signature = parsed.signature;
  return command;
}

SonicControlCommand parse_sonic_command_packet(
    const std::vector<uint8_t>& packet, const std::string& topic,
    std::string* layout_signature) {
  const ParsedPacket parsed = parse_packet_header(packet, topic);
  const bool has_heading = parsed.fields.find("delta_heading") != parsed.fields.end();
  const std::vector<std::string> expected = has_heading
      ? std::vector<std::string>{"start", "stop", "planner", "delta_heading"}
      : std::vector<std::string>{"start", "stop", "planner"};
  if (parsed.order != expected) throw std::runtime_error("SONIC command field order changed");
  SonicControlCommand command;
  command.start = read_scalar<uint8_t>(require_field(parsed, "start", "u8", {1})) != 0;
  command.stop = read_scalar<uint8_t>(require_field(parsed, "stop", "u8", {1})) != 0;
  command.planner = read_scalar<uint8_t>(require_field(parsed, "planner", "u8", {1})) != 0;
  if (command.start && command.stop) throw std::runtime_error("SONIC command cannot start and stop together");
  if (has_heading) {
    command.delta_heading = read_scalar<float>(require_field(parsed, "delta_heading", "f32", {1}));
    require_finite(command.delta_heading, "delta_heading");
    command.has_delta_heading = true;
  }
  if (layout_signature) *layout_signature = parsed.signature;
  return command;
}

bool sonic_planner_mode_is_static(int64_t mode) {
  return mode == 0 || mode == 4 || mode == 5 || mode == 6 || mode == 7 || mode == 9;
}

double sonic_planner_replan_period_sec(int64_t mode) {
  if (mode == static_cast<int64_t>(SonicLocomotionMode::kRun)) return 0.1;
  if (mode == static_cast<int64_t>(SonicLocomotionMode::kCrawling)) return 0.2;
  return 1.0;
}

const char* sonic_planner_mode_name(int64_t mode) {
  static constexpr const char* names[kSonicPlannerModeCountV2] = {
      "idle", "slow_walk", "walk", "run", "squat", "kneel_two_legs", "kneel",
      "lying_face_down", "crawling", "idle_boxing", "walk_boxing", "left_punch",
      "right_punch", "random_punch", "elbow_crawling", "left_hook", "right_hook",
      "forward_jump", "stealth_walk", "injured_walk", "ledge_walking", "object_carrying",
      "stealth_walk_2", "happy_dance_walk", "zombie_walk", "gun_walk", "scare_walk"};
  return mode >= 0 && mode < kSonicPlannerModeCountV2 ? names[mode] : "invalid";
}

SonicPlanner::SonicPlanner(SonicPlannerConfig config) : config_(std::move(config)) {
  if (config_.version != 2 || config_.lookahead_steps < 0 || config_.blend_frames <= 0 ||
      config_.min_output_frames < 2 || config_.max_output_frames > kSonicPlannerMaxFrames ||
      config_.min_output_frames > config_.max_output_frames || !std::isfinite(config_.default_height)) {
    throw std::runtime_error("invalid SONIC planner configuration");
  }
}

std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim>
SonicPlanner::initial_context(const std::vector<float>& hardware_joint_positions) const {
  if (hardware_joint_positions.size() != kSonicMotorCount) {
    throw std::runtime_error("SONIC planner initial joint vector must have 29 elements");
  }
  std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim> context{};
  for (int frame = 0; frame < kSonicPlannerContextFrames; ++frame) {
    const size_t base = static_cast<size_t>(frame * kSonicPlannerQposDim);
    context[base + 2] = config_.default_height;
    context[base + 3] = 1.f;
    for (int joint = 0; joint < kSonicMotorCount; ++joint) {
      require_finite(hardware_joint_positions[joint], "initial joint position");
      context[base + 7 + joint] = hardware_joint_positions[joint];
    }
  }
  return context;
}

std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim>
SonicPlanner::context_from_motion() const {
  if (motion_.empty()) throw std::runtime_error("SONIC planner cannot build context without motion");
  std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim> context{};
  const float generation_frame = static_cast<float>(current_frame_ + config_.lookahead_steps);
  for (int n = 0; n < kSonicPlannerContextFrames; ++n) {
    const float sample_frame = generation_frame + static_cast<float>(n) * (50.f / 30.f);
    const auto sample = sample_motion(motion_, sample_frame);
    const size_t base = static_cast<size_t>(n * kSonicPlannerQposDim);
    for (int i = 0; i < 3; ++i) context[base + i] = sample.root_position[i];
    for (int i = 0; i < 4; ++i) context[base + 3 + i] = sample.body_quat_wxyz[i];
    for (int policy_joint = 0; policy_joint < kSonicMotorCount; ++policy_joint) {
      const int hardware_joint = sonic_policy_to_hardware()[policy_joint];
      context[base + 7 + hardware_joint] = sample.joint_pos[policy_joint];
    }
  }
  return context;
}

std::vector<SonicPlannerMotionFrame> SonicPlanner::run(
    OnnxRunner& runner,
    const std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim>& context,
    const SonicPlannerCommand& command) const {
  using Input = OnnxRunner::TensorInput;
  using Values = OnnxRunner::TensorValues;
  std::unordered_map<std::string, Input> inputs;
  inputs.emplace("context_mujoco_qpos", Input{Values{std::vector<float>(context.begin(), context.end())}, {1, 4, 36}});
  inputs.emplace("target_vel", Input{Values{std::vector<float>{command.speed}}, {1}});
  inputs.emplace("mode", Input{Values{std::vector<int64_t>{command.mode}}, {1}});
  inputs.emplace("movement_direction", Input{Values{std::vector<float>(command.movement.begin(), command.movement.end())}, {1, 3}});
  inputs.emplace("facing_direction", Input{Values{std::vector<float>(command.facing.begin(), command.facing.end())}, {1, 3}});
  inputs.emplace("random_seed", Input{Values{std::vector<int64_t>{config_.random_seed}}, {1}});
  inputs.emplace("has_specific_target", Input{Values{std::vector<int64_t>{0}}, {1, 1}});
  inputs.emplace("specific_target_positions", Input{Values{std::vector<float>(12, 0.f)}, {1, 4, 3}});
  inputs.emplace("specific_target_headings", Input{Values{std::vector<float>(4, 0.f)}, {1, 4}});
  inputs.emplace("allowed_pred_num_tokens", Input{Values{std::vector<int64_t>(
      config_.allowed_pred_num_tokens.begin(), config_.allowed_pred_num_tokens.end())}, {1, 11}});
  inputs.emplace("height", Input{Values{std::vector<float>{command.height}}, {1}});

  const auto outputs = runner.run_named_typed(
      inputs,
      {{"mujoco_qpos", {OnnxRunner::TensorElementType::kFloat32, {1, 64, 36}}},
       {"num_pred_frames", {OnnxRunner::TensorElementType::kInt32, {1}}}});
  const auto& qpos = std::get<std::vector<float>>(outputs.at("mujoco_qpos").values);
  const auto& count = std::get<std::vector<int32_t>>(outputs.at("num_pred_frames").values);
  return resample_output(qpos, count.at(0));
}

std::vector<SonicPlannerMotionFrame> SonicPlanner::resample_output(
    const std::vector<float>& qpos, int32_t num_pred_frames) const {
  if (num_pred_frames < config_.min_output_frames || num_pred_frames > config_.max_output_frames) {
    throw std::runtime_error("SONIC planner num_pred_frames is outside configured [" +
                             std::to_string(config_.min_output_frames) + "," +
                             std::to_string(config_.max_output_frames) + "]");
  }
  if (qpos.size() != static_cast<size_t>(kSonicPlannerMaxFrames * kSonicPlannerQposDim)) {
    throw std::runtime_error("SONIC planner mujoco_qpos output must be [1,64,36]");
  }
  const int output_frames = static_cast<int>(std::floor(
      static_cast<double>(num_pred_frames) / 30.0 * 50.0));
  std::vector<SonicPlannerMotionFrame> result(static_cast<size_t>(output_frames));
  for (int frame = 0; frame < output_frames; ++frame) {
    const float source = static_cast<float>(frame) * (30.f / 50.f);
    const int f0 = static_cast<int>(std::floor(source));
    const int f1 = std::min(f0 + 1, num_pred_frames - 1);
    const float w1 = source - static_cast<float>(f0);
    const float w0 = 1.f - w1;
    auto at = [&](int f, int offset) { return qpos[static_cast<size_t>(f * kSonicPlannerQposDim + offset)]; };
    for (int i = 0; i < 3; ++i) result[frame].root_position[i] = w0 * at(f0, i) + w1 * at(f1, i);
    Quat q0{at(f0, 3), at(f0, 4), at(f0, 5), at(f0, 6)};
    Quat q1{at(f1, 3), at(f1, 4), at(f1, 5), at(f1, 6)};
    result[frame].body_quat_wxyz = slerp(q0, q1, w1);
    for (int policy_joint = 0; policy_joint < kSonicMotorCount; ++policy_joint) {
      const int hardware_joint = sonic_policy_to_hardware()[policy_joint];
      result[frame].joint_pos[policy_joint] =
          w0 * at(f0, 7 + hardware_joint) + w1 * at(f1, 7 + hardware_joint);
    }
  }
  for (int frame = 0; frame + 1 < output_frames; ++frame) {
    for (int joint = 0; joint < kSonicMotorCount; ++joint) {
      result[frame].joint_vel[joint] =
          (result[frame + 1].joint_pos[joint] - result[frame].joint_pos[joint]) * 50.f;
    }
  }
  result.back().joint_vel = result[result.size() - 2].joint_vel;
  return result;
}

void SonicPlanner::initialize(OnnxRunner& runner,
                              const std::vector<float>& hardware_joint_positions) {
  SonicPlannerCommand idle;
  idle.speed = -1.f;
  idle.height = -1.f;
  motion_ = run(runner, initial_context(hardware_joint_positions), idle);
  current_frame_ = 0;
  generated_frame_ = 0;
  last_command_ = idle;
  last_plan_time_ = std::chrono::steady_clock::now();
  initialized_ = true;
}

bool SonicPlanner::should_replan(const SonicPlannerCommand& command,
                                 std::chrono::steady_clock::time_point now) const {
  if (!initialized_) return true;
  if (command.mode != last_command_.mode || !same_vector(command.facing, last_command_.facing) ||
      !near(command.height, last_command_.height)) return true;
  if (sonic_planner_mode_is_static(command.mode)) return false;
  if (!near(command.speed, last_command_.speed) ||
      !same_vector(command.movement, last_command_.movement)) return true;
  return command.speed != 0.f &&
      std::chrono::duration<double>(now - last_plan_time_).count() >=
          sonic_planner_replan_period_sec(command.mode);
}

void SonicPlanner::replan(OnnxRunner& runner, const SonicPlannerCommand& command,
                          std::chrono::steady_clock::time_point now) {
  if (command.mode < 0 || command.mode >= kSonicPlannerModeCountV2) {
    throw std::runtime_error("SONIC planner mode must be in [0,26]");
  }
  if (!initialized_) throw std::runtime_error("SONIC planner must be initialized before replanning");
  generated_frame_ = current_frame_ + config_.lookahead_steps;
  auto generated = run(runner, context_from_motion(), command);
  merge_generated(std::move(generated), generated_frame_);
  last_command_ = command;
  last_plan_time_ = now;
}

void SonicPlanner::merge_generated(std::vector<SonicPlannerMotionFrame> generated,
                                   int generated_frame) {
  if (generated.empty()) throw std::runtime_error("SONIC planner generated empty motion");
  if (motion_.empty()) {
    motion_ = std::move(generated);
    current_frame_ = 0;
    return;
  }
  const int new_length = generated_frame - current_frame_ + static_cast<int>(generated.size());
  if (new_length <= 0) throw std::runtime_error("SONIC planner output arrived after its usable window");
  const int blend_start = std::max(0, generated_frame - current_frame_);
  const auto old = motion_;
  std::vector<SonicPlannerMotionFrame> merged(static_cast<size_t>(new_length));
  for (int frame = 0; frame < new_length; ++frame) {
    const int old_frame = std::clamp(frame + current_frame_, 0, static_cast<int>(old.size()) - 1);
    const int new_frame = std::clamp(frame + current_frame_ - generated_frame, 0,
                                     static_cast<int>(generated.size()) - 1);
    const float w_new = std::clamp(
        static_cast<float>(frame - blend_start) / static_cast<float>(config_.blend_frames), 0.f, 1.f);
    const float w_old = 1.f - w_new;
    for (int i = 0; i < 3; ++i) {
      merged[frame].root_position[i] = w_old * old[old_frame].root_position[i] +
                                      w_new * generated[new_frame].root_position[i];
    }
    merged[frame].body_quat_wxyz = slerp(old[old_frame].body_quat_wxyz,
                                         generated[new_frame].body_quat_wxyz, w_new);
    for (int joint = 0; joint < kSonicMotorCount; ++joint) {
      merged[frame].joint_pos[joint] = w_old * old[old_frame].joint_pos[joint] +
                                      w_new * generated[new_frame].joint_pos[joint];
      merged[frame].joint_vel[joint] = w_old * old[old_frame].joint_vel[joint] +
                                      w_new * generated[new_frame].joint_vel[joint];
    }
  }
  motion_ = std::move(merged);
  current_frame_ = 0;
}

void SonicPlanner::advance() {
  if (!motion_.empty()) current_frame_ = std::min(current_frame_ + 1, static_cast<int>(motion_.size()) - 1);
}

std::vector<SonicReferenceFrame> SonicPlanner::reference_frames() const {
  std::vector<SonicReferenceFrame> result(motion_.size());
  for (size_t frame = 0; frame < motion_.size(); ++frame) {
    result[frame].joint_pos = motion_[frame].joint_pos;
    result[frame].joint_vel = motion_[frame].joint_vel;
    result[frame].body_quat_wxyz = motion_[frame].body_quat_wxyz;
    result[frame].root_z = motion_[frame].root_position[2];
    result[frame].frame_index = static_cast<int64_t>(frame);
    result[frame].has_explicit_frame_index = true;
    result[frame].encoder_mode = 0;
  }
  return result;
}

}  // namespace action_go::runtime
