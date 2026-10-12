#include "action_go/runtime/sonic_policy.h"

#include "action_go/runtime/policy_protocol.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstring>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace action_go::runtime {
namespace {

using Quat = std::array<float, 4>;

std::vector<std::vector<float>> read_csv(const std::filesystem::path& path, size_t width) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("SONIC CSV not found: " + path.string());
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("SONIC CSV is empty: " + path.string());
  std::vector<std::vector<float>> rows;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::stringstream stream(line);
    std::string cell;
    std::vector<float> row;
    while (std::getline(stream, cell, ',')) {
      const float value = std::stof(cell);
      if (!std::isfinite(value)) throw std::runtime_error("SONIC CSV contains NaN/Inf: " + path.string());
      row.push_back(value);
    }
    if (row.size() != width) {
      throw std::runtime_error("SONIC CSV width mismatch in " + path.string() +
                               ": expected " + std::to_string(width) +
                               ", got " + std::to_string(row.size()));
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) throw std::runtime_error("SONIC CSV has no data rows: " + path.string());
  return rows;
}

Quat normalize(Quat q) {
  float norm = 0.f;
  for (float value : q) norm += value * value;
  norm = std::sqrt(norm);
  if (!std::isfinite(norm) || norm < 1e-8f) throw std::runtime_error("SONIC quaternion is invalid");
  for (float& value : q) value /= norm;
  return q;
}

Quat conjugate(const Quat& q) { return {q[0], -q[1], -q[2], -q[3]}; }

Quat multiply(const Quat& a, const Quat& b) {
  return normalize({a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]});
}

Quat heading(const Quat& raw) {
  const Quat q = normalize(raw);
  const float yaw = std::atan2(2.f * (q[0] * q[3] + q[1] * q[2]),
                               1.f - 2.f * (q[2] * q[2] + q[3] * q[3]));
  return {std::cos(0.5f * yaw), 0.f, 0.f, std::sin(0.5f * yaw)};
}

std::array<float, 6> rotation_6d(const Quat& raw) {
  const Quat q = normalize(raw);
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  return {1.f - 2.f * (y * y + z * z), 2.f * (x * y - z * w),
          2.f * (x * y + z * w), 1.f - 2.f * (x * x + z * z),
          2.f * (x * z - y * w), 2.f * (y * z + x * w)};
}

std::array<float, 3> gravity_body(const Quat& raw) {
  const Quat q = normalize(raw);
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  return {2.f * (-z * x + w * y), -2.f * (z * y + w * x),
          1.f - 2.f * (w * w + z * z)};
}

void append(std::vector<float>& output, const float* values, size_t count) {
  output.insert(output.end(), values, values + count);
}

template <typename T>
T byte_swap(T value) {
  std::array<uint8_t, sizeof(T)> source{};
  std::array<uint8_t, sizeof(T)> target{};
  std::memcpy(source.data(), &value, sizeof(T));
  std::reverse_copy(source.begin(), source.end(), target.begin());
  std::memcpy(&value, target.data(), sizeof(T));
  return value;
}

bool native_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

}  // namespace

const std::array<int, kSonicMotorCount>& sonic_policy_to_hardware() {
  static const std::array<int, kSonicMotorCount> map{
      0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
      16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};
  return map;
}

const std::array<int, kSonicMotorCount>& sonic_hardware_to_policy() {
  static const std::array<int, kSonicMotorCount> map{
      0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
      11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
  return map;
}

std::array<float, kSonicMotorCount> sonic_robot_to_policy(
    const std::vector<float>& hardware_values,
    const std::vector<float>& default_q,
    bool subtract_default) {
  if (hardware_values.size() < kSonicMotorCount || default_q.size() != kSonicMotorCount) {
    throw std::runtime_error("SONIC robot vector size mismatch");
  }
  std::array<float, kSonicMotorCount> output{};
  for (int policy = 0; policy < kSonicMotorCount; ++policy) {
    const int hardware = sonic_policy_to_hardware()[policy];
    output[policy] = hardware_values[hardware] - (subtract_default ? default_q[hardware] : 0.f);
  }
  return output;
}

std::vector<float> sonic_decode_target(const std::vector<float>& policy_action,
                                       const std::vector<float>& default_q,
                                       const std::vector<float>& action_scale) {
  if (policy_action.size() != kSonicMotorCount || default_q.size() != kSonicMotorCount ||
      action_scale.size() != kSonicMotorCount) {
    throw std::runtime_error("SONIC action decoder vector size mismatch");
  }
  std::vector<float> output(kSonicMotorCount);
  for (int hardware = 0; hardware < kSonicMotorCount; ++hardware) {
    const int policy = sonic_hardware_to_policy()[hardware];
    const float action = policy_action[policy];
    if (!std::isfinite(action)) throw std::runtime_error("SONIC action contains NaN/Inf");
    output[hardware] = default_q[hardware] + action * action_scale[hardware];
  }
  return output;
}

std::vector<float> sonic_startup_target(const std::vector<float>& measured_q,
                                        const std::vector<float>& default_q,
                                        float alpha) {
  if (measured_q.size() != kSonicMotorCount || default_q.size() != kSonicMotorCount) {
    throw std::runtime_error("SONIC startup target dimension mismatch");
  }
  if (!std::isfinite(alpha)) throw std::runtime_error("SONIC startup alpha is NaN/Inf");
  alpha = std::clamp(alpha, 0.f, 1.f);
  std::vector<float> target(kSonicMotorCount);
  for (int i = 0; i < kSonicMotorCount; ++i) {
    if (!std::isfinite(measured_q[i]) || !std::isfinite(default_q[i])) {
      throw std::runtime_error("SONIC startup pose contains NaN/Inf");
    }
    target[i] = (1.f - alpha) * measured_q[i] + alpha * default_q[i];
  }
  return target;
}

std::vector<SonicReferenceFrame> parse_sonic_zmq_packet(
    const std::vector<uint8_t>& packet, const std::string& topic,
    std::string* layout_signature) {
  size_t offset = 0;
  if (!topic.empty()) {
    if (packet.size() < topic.size() ||
        std::memcmp(packet.data(), topic.data(), topic.size()) != 0) {
      throw std::runtime_error("SONIC ZMQ topic prefix mismatch");
    }
    offset = topic.size();
  }
  if (packet.size() < offset + kSonicZmqHeaderSize) {
    throw std::runtime_error("SONIC ZMQ packet is shorter than the 1280-byte header");
  }
  const char* header_data = reinterpret_cast<const char*>(packet.data() + offset);
  const size_t header_length = strnlen(header_data, kSonicZmqHeaderSize);
  const auto header = nlohmann::json::parse(std::string(header_data, header_length));
  const int protocol_version = header.value("v", 0);
  if (protocol_version < 1 || protocol_version > 3) {
    throw std::runtime_error("SONIC ZMQ pose protocol version must be 1, 2, or 3");
  }
  if (!header.contains("count")) throw std::runtime_error("SONIC ZMQ header is missing count");
  const std::string endian = header.value("endian", std::string("le"));
  if (endian != "le" && endian != "be") throw std::runtime_error("SONIC ZMQ endian must be le or be");
  const bool data_little = endian == "le";
  const bool swap = data_little != native_little_endian();

  struct FieldView {
    std::string dtype;
    std::vector<size_t> shape;
    const uint8_t* data{nullptr};
    size_t bytes{0};
  };
  std::unordered_map<std::string, FieldView> fields;
  std::ostringstream signature;
  signature << "v" << protocol_version << ":" << endian;
  size_t data_offset = offset + kSonicZmqHeaderSize;
  for (const auto& field : header.at("fields")) {
    FieldView view;
    const std::string name = field.at("name").get<std::string>();
    view.dtype = field.at("dtype").get<std::string>();
    view.shape = field.at("shape").get<std::vector<size_t>>();
    signature << ";" << name << ":" << view.dtype << ":[N";
    for (size_t i = 1; i < view.shape.size(); ++i) signature << "," << view.shape[i];
    signature << "]";
    size_t elements = 1;
    for (size_t dim : view.shape) {
      if (dim != 0 && elements > std::numeric_limits<size_t>::max() / dim) {
        throw std::runtime_error("SONIC ZMQ field shape overflows size_t: " + name);
      }
      elements *= dim;
    }
    const size_t element_size = view.dtype == "f64" || view.dtype == "i64" ? 8
                                : view.dtype == "bool" || view.dtype == "u8" ? 1 : 4;
    if (view.dtype != "f32" && view.dtype != "f64" &&
        view.dtype != "i32" && view.dtype != "i64" &&
        view.dtype != "bool" && view.dtype != "u8") {
      throw std::runtime_error("SONIC ZMQ unsupported dtype: " + view.dtype);
    }
    if (elements > std::numeric_limits<size_t>::max() / element_size) {
      throw std::runtime_error("SONIC ZMQ field byte size overflows size_t: " + name);
    }
    view.bytes = elements * element_size;
    if (data_offset > packet.size() || view.bytes > packet.size() - data_offset) {
      throw std::runtime_error("SONIC ZMQ field exceeds packet bounds: " + name);
    }
    view.data = packet.data() + data_offset;
    data_offset += view.bytes;
    if (!fields.emplace(name, std::move(view)).second) {
      throw std::runtime_error("SONIC ZMQ duplicate field: " + name);
    }
  }
  const auto quaternion_field = fields.find("body_quat_w") != fields.end()
                                    ? fields.find("body_quat_w") : fields.find("body_quat");
  if (quaternion_field == fields.end()) {
    throw std::runtime_error("SONIC ZMQ missing required field: body_quat_w (or body_quat)");
  }
  std::vector<const char*> required{"joint_pos", "joint_vel"};
  if (protocol_version == 2) required.clear();
  if (protocol_version >= 2) {
    required.push_back("smpl_joints");
    required.push_back("smpl_pose");
    required.push_back("frame_index");
  }
  for (const char* required_name : required) {
    if (fields.find(required_name) == fields.end()) {
      throw std::runtime_error(std::string("SONIC ZMQ missing required field: ") + required_name);
    }
  }
  const FieldView* positions = fields.find("joint_pos") == fields.end() ? nullptr : &fields.at("joint_pos");
  const FieldView* velocities = fields.find("joint_vel") == fields.end() ? nullptr : &fields.at("joint_vel");
  const FieldView* smpl_joints = fields.find("smpl_joints") == fields.end() ? nullptr : &fields.at("smpl_joints");
  const FieldView* smpl_pose = fields.find("smpl_pose") == fields.end() ? nullptr : &fields.at("smpl_pose");
  const auto& quaternions = quaternion_field->second;
  if ((protocol_version >= 2 && (!smpl_joints || !smpl_pose)) ||
      (protocol_version == 1 && (!positions || !velocities))) {
    throw std::runtime_error("SONIC ZMQ primary reference fields are missing");
  }
  const auto& primary = protocol_version >= 2 ? fields.at("smpl_joints") : fields.at("joint_pos");
  if (primary.shape.empty()) {
    throw std::runtime_error("SONIC ZMQ primary reference field has an empty shape");
  }
  const size_t frames = primary.shape[0];
  if (positions && (positions->shape.size() != 2 || positions->shape[1] != kSonicMotorCount ||
                    !velocities || velocities->shape != positions->shape || positions->shape[0] != frames)) {
    throw std::runtime_error("SONIC ZMQ joint_pos/joint_vel must have matching [N,29] shapes");
  }
  if (protocol_version >= 2 &&
      (smpl_joints->shape != std::vector<size_t>{frames, kSonicSmplJointCount, 3} ||
       smpl_pose->shape != std::vector<size_t>{frames, kSonicSmplPoseCount, 3})) {
    throw std::runtime_error("SONIC ZMQ v2/v3 smpl_joints/smpl_pose must be [N,24,3]/[N,21,3]");
  }
  if (frames == 0 ||
      !((quaternions.shape.size() == 2 && quaternions.shape[0] == frames && quaternions.shape[1] == 4) ||
        (quaternions.shape.size() == 3 && quaternions.shape[0] == frames && quaternions.shape[1] == 1 &&
         quaternions.shape[2] == 4))) {
    throw std::runtime_error("SONIC ZMQ body_quat must have shape [N,4] or [N,1,4]");
  }

  auto read_float = [&](const FieldView& field, size_t index) {
    if (field.dtype == "f32") {
      float value;
      std::memcpy(&value, field.data + index * sizeof(float), sizeof(float));
      if (swap) value = byte_swap(value);
      return value;
    }
    if (field.dtype == "f64") {
      double value;
      std::memcpy(&value, field.data + index * sizeof(double), sizeof(double));
      if (swap) value = byte_swap(value);
      return static_cast<float>(value);
    }
    throw std::runtime_error("SONIC ZMQ floating field must be f32 or f64");
  };
  auto read_integer = [&](const FieldView& field, size_t index) -> int64_t {
    if (field.dtype == "i32") {
      int32_t value;
      std::memcpy(&value, field.data + index * sizeof(value), sizeof(value));
      return swap ? byte_swap(value) : value;
    }
    if (field.dtype == "i64") {
      int64_t value;
      std::memcpy(&value, field.data + index * sizeof(value), sizeof(value));
      return swap ? byte_swap(value) : value;
    }
    throw std::runtime_error("SONIC ZMQ frame_index must be i32 or i64");
  };

  std::vector<SonicReferenceFrame> output(frames);
  const size_t quaternion_bodies = quaternions.shape.size() == 3 ? quaternions.shape[1] : 1;
  const auto frame_index = fields.find("frame_index");
  if (frame_index != fields.end() &&
      (frame_index->second.shape.size() != 1 || frame_index->second.shape[0] != frames)) {
    throw std::runtime_error("SONIC ZMQ frame_index must have shape [N]");
  }
  for (size_t frame = 0; frame < frames; ++frame) {
    output[frame].encoder_mode = protocol_version == 1 ? 0 : 2;
    if (positions) {
      for (size_t joint = 0; joint < kSonicMotorCount; ++joint) {
        output[frame].joint_pos[joint] = read_float(*positions, frame * kSonicMotorCount + joint);
        output[frame].joint_vel[joint] = read_float(*velocities, frame * kSonicMotorCount + joint);
      }
    }
    if (smpl_joints) {
      for (size_t value = 0; value < output[frame].smpl_joints.size(); ++value) {
        output[frame].smpl_joints[value] =
            read_float(*smpl_joints, frame * output[frame].smpl_joints.size() + value);
      }
    }
    for (size_t component = 0; component < 4; ++component) {
      output[frame].body_quat_wxyz[component] =
          read_float(quaternions, frame * quaternion_bodies * 4 + component);
    }
    output[frame].body_quat_wxyz = normalize(output[frame].body_quat_wxyz);
    output[frame].frame_index = frame_index == fields.end()
                                    ? static_cast<int64_t>(frame)
                                    : read_integer(frame_index->second, frame);
    output[frame].has_explicit_frame_index = frame_index != fields.end();
    if (frame > 0 && output[frame].frame_index <= output[frame - 1].frame_index) {
      throw std::runtime_error("SONIC ZMQ frame_index must increase monotonically");
    }
    if (frame > 1 && output[frame].frame_index - output[frame - 1].frame_index !=
                         output[1].frame_index - output[0].frame_index) {
      throw std::runtime_error("SONIC ZMQ frame_index stride must be constant");
    }
  }
  if (data_offset != packet.size()) throw std::runtime_error("SONIC ZMQ packet has trailing bytes");
  if (layout_signature) *layout_signature = signature.str();
  return output;
}

std::vector<SonicReferenceFrame> parse_sonic_zmq_v1_packet(
    const std::vector<uint8_t>& packet, const std::string& topic,
    std::string* layout_signature) {
  return parse_sonic_zmq_packet(packet, topic, layout_signature);
}

void SonicReferenceCache::load_csv_directory(const std::filesystem::path& directory) {
  const auto positions = read_csv(directory / "joint_pos.csv", kSonicMotorCount);
  const auto velocities = read_csv(directory / "joint_vel.csv", kSonicMotorCount);
  const auto quaternions = read_csv(directory / "body_quat.csv", 56);
  const auto body_positions = read_csv(directory / "body_pos.csv", 42);
  if (positions.size() != velocities.size() || positions.size() != quaternions.size() ||
      positions.size() != body_positions.size()) {
    throw std::runtime_error("SONIC reference CSV row counts differ");
  }
  std::vector<SonicReferenceFrame> frames(positions.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    std::copy_n(positions[i].begin(), kSonicMotorCount, frames[i].joint_pos.begin());
    std::copy_n(velocities[i].begin(), kSonicMotorCount, frames[i].joint_vel.begin());
    std::copy_n(quaternions[i].begin(), 4, frames[i].body_quat_wxyz.begin());
    frames[i].body_quat_wxyz = normalize(frames[i].body_quat_wxyz);
    frames[i].root_z = body_positions[i][2];
    frames[i].frame_index = static_cast<int64_t>(i);
  }
  replace(std::move(frames));
}

void SonicReferenceCache::replace(std::vector<SonicReferenceFrame> frames) {
  if (frames.empty()) throw std::runtime_error("SONIC reference contains no frames");
  for (size_t i = 0; i < frames.size(); ++i) {
    frames[i].body_quat_wxyz = normalize(frames[i].body_quat_wxyz);
    if (frames[i].encoder_mode != frames.front().encoder_mode) {
      throw std::runtime_error("SONIC reference changes encoder mode within one window");
    }
    if (i > 0 && frames[i].frame_index <= frames[i - 1].frame_index) {
      throw std::runtime_error("SONIC reference frame_index must increase monotonically");
    }
  }
  frames_ = std::move(frames);
}

void SonicReferenceCache::append(std::vector<SonicReferenceFrame> frames) {
  if (frames.empty()) return;
  if (frames_.empty()) {
    replace(std::move(frames));
    return;
  }
  for (auto& frame : frames) frame.body_quat_wxyz = normalize(frame.body_quat_wxyz);
  if (frames.front().encoder_mode != frames_.front().encoder_mode) {
    throw std::runtime_error("SONIC reference encoder mode changed while merging a stream");
  }
  if (frames.front().frame_index <= frames_.back().frame_index) {
    throw std::runtime_error("SONIC reference append is not newer than the cached window");
  }
  frames_.insert(frames_.end(), std::make_move_iterator(frames.begin()),
                 std::make_move_iterator(frames.end()));
}

void SonicReferenceCache::merge_tail(std::vector<SonicReferenceFrame> frames) {
  if (frames.empty()) return;
  if (frames_.empty()) {
    replace(std::move(frames));
    return;
  }
  const auto overlap = std::lower_bound(frames_.begin(), frames_.end(), frames.front().frame_index,
                                        [](const SonicReferenceFrame& frame, int64_t index) {
                                          return frame.frame_index < index;
                                        });
  frames_.erase(overlap, frames_.end());
  append(std::move(frames));
}

void SonicReferenceCache::discard_before(int64_t frame_index) {
  const auto keep = std::lower_bound(frames_.begin(), frames_.end(), frame_index,
                                     [](const SonicReferenceFrame& frame, int64_t index) {
                                       return frame.frame_index < index;
                                     });
  frames_.erase(frames_.begin(), keep);
}

const SonicReferenceFrame& SonicReferenceCache::frame_clamped(int64_t index) const {
  if (frames_.empty()) throw std::runtime_error("SONIC reference is empty");
  return frames_[static_cast<size_t>(std::clamp<int64_t>(index, 0, static_cast<int64_t>(frames_.size() - 1)))];
}

int64_t SonicReferenceCache::first_frame_index() const {
  if (frames_.empty()) throw std::runtime_error("SONIC reference is empty");
  return frames_.front().frame_index;
}

int64_t SonicReferenceCache::last_frame_index() const {
  if (frames_.empty()) throw std::runtime_error("SONIC reference is empty");
  return frames_.back().frame_index;
}

int64_t SonicReferenceCache::frame_step() const {
  if (frames_.size() < 2) return 1;
  return frames_[1].frame_index - frames_[0].frame_index;
}

int64_t SonicReferenceCache::local_index_for_frame(int64_t frame_index) const {
  if (frames_.empty()) throw std::runtime_error("SONIC reference is empty");
  const auto found = std::lower_bound(frames_.begin(), frames_.end(), frame_index,
                                      [](const SonicReferenceFrame& frame, int64_t index) {
                                        return frame.frame_index < index;
                                      });
  if (found == frames_.end() || found->frame_index != frame_index) {
    throw std::runtime_error("SONIC reference frame is outside the cached window");
  }
  return static_cast<int64_t>(std::distance(frames_.begin(), found));
}

void SonicObservationBuilder::reset(const SonicRobotFrame& initial_robot,
                                    const SonicReferenceFrame& initial_reference) {
  history_.clear();
  reset_heading(initial_robot, initial_reference);
}

void SonicObservationBuilder::reset_heading(const SonicRobotFrame& robot,
                                            const SonicReferenceFrame& reference) {
  initial_robot_heading_ = heading(robot.base_quat_wxyz);
  initial_reference_heading_ = heading(reference.body_quat_wxyz);
}

void SonicObservationBuilder::push_robot_frame(const SonicRobotFrame& frame) {
  history_.insert(history_.begin(), frame);
  if (history_.size() > kSonicHistoryLen) history_.resize(kSonicHistoryLen);
}

std::vector<float> SonicObservationBuilder::build_encoder_observation(
    const SonicReferenceCache& reference, int64_t current_frame,
    int future_frames, int future_step) const {
  if (history_.empty()) throw std::runtime_error("SONIC robot history is empty");
  const bool v11 = layout_ == SonicObservationLayout::kV11;
  const size_t encoder_obs_dim = v11 ? kSonicV11EncoderObsDim : kSonicEncoderObsDim;
  std::vector<float> output(encoder_obs_dim, 0.f);
  const int encoder_mode = reference.frame_clamped(current_frame).encoder_mode;
  output[0] = static_cast<float>(encoder_mode);
  const Quat apply_heading = multiply(initial_robot_heading_, conjugate(initial_reference_heading_));
  const Quat orientation_inverse = v11
      ? conjugate(heading(history_.front().base_quat_wxyz))
      : conjugate(normalize(history_.front().base_quat_wxyz));
  if (encoder_mode == 0) {
    if (future_frames != kSonicReferenceFrames || future_step != kSonicReferenceStep) {
      throw std::runtime_error("SONIC G1 reference horizon does not match released checkpoint");
    }
    size_t offset = 4;
    for (int i = 0; i < future_frames; ++i) {
      const auto& frame = reference.frame_clamped(current_frame + i * future_step);
      std::copy(frame.joint_pos.begin(), frame.joint_pos.end(), output.begin() + offset);
      offset += kSonicMotorCount;
    }
    for (int i = 0; i < future_frames; ++i) {
      const auto& frame = reference.frame_clamped(current_frame + i * future_step);
      std::copy(frame.joint_vel.begin(), frame.joint_vel.end(), output.begin() + offset);
      offset += kSonicMotorCount;
    }
    if (!v11) {
      offset += kSonicReferenceFrames + 1 + 6;
    }
    for (int i = 0; i < future_frames; ++i) {
      const auto& frame = reference.frame_clamped(current_frame + i * future_step);
      const Quat aligned_reference = multiply(apply_heading, frame.body_quat_wxyz);
      const auto orientation = rotation_6d(multiply(orientation_inverse, aligned_reference));
      std::copy(orientation.begin(), orientation.end(), output.begin() + offset);
      offset += orientation.size();
    }
  } else if (encoder_mode == 2) {
    size_t offset = v11 ? 911 : 922;
    for (int i = 0; i < kSonicSmplFrames; ++i) {
      const auto& frame = reference.frame_clamped(current_frame + i);
      std::copy(frame.smpl_joints.begin(), frame.smpl_joints.end(), output.begin() + offset);
      offset += frame.smpl_joints.size();
    }
    offset = v11 ? 1631 : 1642;
    for (int i = 0; i < kSonicSmplFrames; ++i) {
      const auto& frame = reference.frame_clamped(current_frame + i);
      const Quat aligned_reference = multiply(apply_heading, frame.body_quat_wxyz);
      const auto orientation = rotation_6d(multiply(orientation_inverse, aligned_reference));
      std::copy(orientation.begin(), orientation.end(), output.begin() + offset);
      offset += orientation.size();
    }
    static constexpr std::array<int, 6> kWristPolicyIndices{23, 24, 25, 26, 27, 28};
    offset = v11 ? 1691 : 1702;
    for (int i = 0; i < kSonicSmplFrames; ++i) {
      const auto& frame = reference.frame_clamped(current_frame + i);
      for (int joint : kWristPolicyIndices) output[offset++] = frame.joint_pos[joint];
    }
  } else {
    throw std::runtime_error("SONIC reference uses unsupported encoder mode " +
                             std::to_string(encoder_mode));
  }
  if (output.size() != encoder_obs_dim) throw std::runtime_error("SONIC encoder observation size mismatch");
  return output;
}

std::vector<float> SonicObservationBuilder::build_decoder_observation(
    const std::vector<float>& token) const {
  if (token.size() != kSonicTokenDim) throw std::runtime_error("SONIC token dimension mismatch");
  if (history_.empty()) throw std::runtime_error("SONIC robot history is empty");
  std::vector<float> output;
  output.reserve(kSonicDecoderObsDim);
  append(output, token.data(), token.size());
  auto frame_at = [&](size_t oldest_first_index) -> const SonicRobotFrame* {
    const size_t reverse_index = kSonicHistoryLen - 1 - oldest_first_index;
    return reverse_index < history_.size() ? &history_[reverse_index] : nullptr;
  };
  for (size_t i = 0; i < kSonicHistoryLen; ++i) {
    if (const auto* frame = frame_at(i)) append(output, frame->base_ang_vel.data(), 3);
    else output.insert(output.end(), 3, 0.f);
  }
  for (size_t i = 0; i < kSonicHistoryLen; ++i) {
    if (const auto* frame = frame_at(i)) append(output, frame->joint_pos.data(), kSonicMotorCount);
    else output.insert(output.end(), kSonicMotorCount, 0.f);
  }
  for (size_t i = 0; i < kSonicHistoryLen; ++i) {
    if (const auto* frame = frame_at(i)) append(output, frame->joint_vel.data(), kSonicMotorCount);
    else output.insert(output.end(), kSonicMotorCount, 0.f);
  }
  for (size_t i = 0; i < kSonicHistoryLen; ++i) {
    if (const auto* frame = frame_at(i)) append(output, frame->last_action.data(), kSonicMotorCount);
    else output.insert(output.end(), kSonicMotorCount, 0.f);
  }
  for (size_t i = 0; i < kSonicHistoryLen; ++i) {
    if (const auto* frame = frame_at(i)) {
      const auto gravity = gravity_body(frame->base_quat_wxyz);
      append(output, gravity.data(), gravity.size());
    } else {
      output.insert(output.end(), 3, 0.f);
    }
  }
  if (output.size() != kSonicDecoderObsDim) throw std::runtime_error("SONIC decoder observation size mismatch");
  return output;
}

}  // namespace action_go::runtime
