#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace action_go::runtime {

struct SuperMimicReferenceFrame {
  std::array<float, 730> goal{};
  std::array<float, 9> root_rotation{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
};

struct SuperMimicRobotFrame {
  std::array<float, 29> q{};
  std::array<float, 29> dq{};
  std::array<float, 4> quat{1.f, 0.f, 0.f, 0.f};
  std::array<float, 3> gyro{};
};

class SuperMimicController {
 public:
  static constexpr std::size_t kActionDim = 29;
  static constexpr std::size_t kObsFrameDim = 99;
  static constexpr std::size_t kHistoryLen = 10;
  static constexpr std::size_t kGoalDim = 730;
  static constexpr std::size_t kObsDim = 1720;
  static constexpr std::size_t kReferenceRowDim = 739;
  static constexpr uint32_t kPolicyTickMs = 20;
  using Rotation = std::array<float, 9>;
  using Observation = std::array<float, kObsFrameDim>;

  static const std::array<const char*, kActionDim>& sdk_joint_names() {
    static const std::array<const char*, kActionDim> names{
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"};
    return names;
  }

  void configure(const std::vector<float>& default_q, const std::vector<float>& action_scale) {
    require_joint_values(default_q, "default angles");
    require_joint_values(action_scale, "action scale");
    std::copy(default_q.begin(), default_q.end(), default_q_.begin());
    std::copy(action_scale.begin(), action_scale.end(), action_scale_.begin());
    configured_ = true;
  }

  static std::vector<SuperMimicReferenceFrame> read_reference_csv(std::istream& input) {
    std::vector<SuperMimicReferenceFrame> frames;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      const std::string context = "SuperMimic reference CSV line " + std::to_string(line_number);
      std::array<float, kReferenceRowDim> values{};
      std::size_t begin = 0;
      for (std::size_t index = 0; index < values.size(); ++index) {
        const auto end = line.find(',', begin);
        if ((index + 1 < values.size()) != (end != std::string::npos)) {
          throw std::runtime_error(context + " must contain exactly 739 comma-separated floats");
        }
        std::istringstream token(line.substr(begin, end == std::string::npos ? end : end - begin));
        token.imbue(std::locale::classic());
        if (!(token >> values[index]) || !std::isfinite(values[index])) {
          throw std::runtime_error(context + " contains an invalid/nonfinite float");
        }
        token >> std::ws;
        if (!token.eof()) throw std::runtime_error(context + " contains a malformed float");
        if (end != std::string::npos) begin = end + 1;
      }
      SuperMimicReferenceFrame frame;
      std::copy_n(values.begin(), kGoalDim, frame.goal.begin());
      std::copy_n(values.begin() + kGoalDim, 9, frame.root_rotation.begin());
      frames.push_back(std::move(frame));
    }
    if (input.bad() || (!input.eof() && input.fail())) {
      throw std::runtime_error("failed to read SuperMimic reference CSV");
    }
    if (frames.empty()) throw std::runtime_error("SuperMimic reference CSV contains no frames");
    return frames;
  }

  void load_reference_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("SuperMimic reference CSV not found: " + path);
    set_reference_frames(read_reference_csv(input));
  }

  void set_reference_frames(std::vector<SuperMimicReferenceFrame> frames) {
    if (frames.empty()) throw std::runtime_error("SuperMimic reference contains no frames");
    for (const auto& frame : frames) {
      require_finite(frame.goal, "reference goal");
      require_finite(frame.root_rotation, "reference rotation");
    }
    reference_ = std::move(frames);
    frame_limit_ = reference_.size();
    reset({1.f, 0.f, 0.f, 0.f}, false);
  }

  void set_frame_limit(std::size_t count) {
    if (count == 0 || count > reference_.size()) {
      throw std::runtime_error("SuperMimic frame limit must be in [1, reference frame count]");
    }
    frame_limit_ = count;
  }

  void reset(const std::array<float, 4>& robot_quat, bool align_initial_yaw) {
    policy_step_ = 0;
    previous_action_.fill(0.f);
    history_initialized_ = false;
    initial_to_world_ = identity_rotation();
    if (align_initial_yaw) {
      if (reference_.empty()) throw std::runtime_error("SuperMimic reference is not loaded");
      const auto robot_rotation = rotation_from_quaternion(robot_quat);
      const auto& ref_rotation = reference_.front().root_rotation;
      const float yaw = std::atan2(robot_rotation[3], robot_rotation[0]) -
                        std::atan2(ref_rotation[3], ref_rotation[0]);
      const float cosine = std::cos(yaw);
      const float sine = std::sin(yaw);
      initial_to_world_ = {cosine, -sine, 0.f, sine, cosine, 0.f, 0.f, 0.f, 1.f};
    }
  }

  std::size_t reference_size() const { return reference_.size(); }
  std::size_t frame_limit() const { return frame_limit_; }
  uint64_t policy_step() const { return policy_step_; }
  bool at_end() const { return policy_step_ >= frame_limit_; }
  std::size_t reference_index() const {
    if (frame_limit_ == 0) throw std::runtime_error("SuperMimic reference is not loaded");
    return static_cast<std::size_t>(std::min<uint64_t>(policy_step_, frame_limit_ - 1));
  }
  const std::array<float, kActionDim>& previous_action() const { return previous_action_; }
  const Rotation& initial_to_world() const { return initial_to_world_; }

  Observation observation(const SuperMimicRobotFrame& robot) const {
    if (!configured_) throw std::runtime_error("SuperMimic controller parameters are not configured");
    require_finite(robot.q, "joint positions");
    require_finite(robot.dq, "joint velocities");
    require_finite(robot.gyro, "pelvis gyro");
    const auto robot_rotation = rotation_from_quaternion(robot.quat);
    const auto aligned_ref = multiply(initial_to_world_, reference_.at(reference_index()).root_rotation);
    const auto relative_ref = multiply(transpose(robot_rotation), aligned_ref);
    Observation obs{};
    for (std::size_t joint = 0; joint < kActionDim; ++joint) {
      obs[joint] = robot.q[joint] - default_q_[joint];
      obs[29 + joint] = robot.dq[joint];
      obs[64 + joint] = previous_action_[joint];
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      obs[58 + axis] = robot.gyro[axis];
      obs[61 + axis] = -robot_rotation[6 + axis];
      obs[93 + 2 * axis] = relative_ref[3 * axis];
      obs[94 + 2 * axis] = relative_ref[3 * axis + 1];
    }
    require_finite(obs, "observation");
    return obs;
  }

  std::vector<float> build_input(const SuperMimicRobotFrame& robot) {
    const auto obs = observation(robot);
    if (!history_initialized_) {
      history_.fill(obs);
      history_initialized_ = true;
    } else {
      for (std::size_t index = 1; index < kHistoryLen; ++index) history_[index - 1] = history_[index];
      history_.back() = obs;
    }
    std::vector<float> input;
    input.reserve(kObsDim);
    for (const auto& frame : history_) input.insert(input.end(), frame.begin(), frame.end());
    const auto& goal = reference_.at(reference_index()).goal;
    input.insert(input.end(), goal.begin(), goal.end());
    return input;
  }

  std::vector<float> decode_action(const std::vector<float>& action) const {
    if (!configured_) throw std::runtime_error("SuperMimic controller parameters are not configured");
    require_joint_values(action, "action");
    std::vector<float> target(kActionDim);
    for (std::size_t joint = 0; joint < kActionDim; ++joint) {
      target[joint] = default_q_[joint] + action_scale_[joint] * action[joint];
    }
    require_finite(target, "joint targets");
    return target;
  }

  void accept_action(const std::vector<float>& action) {
    require_joint_values(action, "previous action");
    if (policy_step_ == std::numeric_limits<uint64_t>::max()) {
      throw std::runtime_error("SuperMimic policy step overflow");
    }
    std::copy(action.begin(), action.end(), previous_action_.begin());
    ++policy_step_;
  }

  static Rotation rotation_from_quaternion(const std::array<float, 4>& quat) {
    require_finite(quat, "pelvis quaternion");
    float norm_squared = 0.f;
    for (float component : quat) norm_squared += component * component;
    if (!std::isfinite(norm_squared) || norm_squared < 1e-12f) {
      throw std::runtime_error("SuperMimic pelvis quaternion has invalid norm");
    }
    const float inverse_norm = 1.f / std::sqrt(norm_squared);
    const float qw = quat[0] * inverse_norm;
    const float qx = quat[1] * inverse_norm;
    const float qy = quat[2] * inverse_norm;
    const float qz = quat[3] * inverse_norm;
    return {1.f - 2.f * (qy * qy + qz * qz), 2.f * (qx * qy - qw * qz), 2.f * (qx * qz + qw * qy),
            2.f * (qx * qy + qw * qz), 1.f - 2.f * (qx * qx + qz * qz), 2.f * (qy * qz - qw * qx),
            2.f * (qx * qz - qw * qy), 2.f * (qy * qz + qw * qx), 1.f - 2.f * (qx * qx + qy * qy)};
  }

 private:
  template <typename Values>
  static void require_finite(const Values& values, const char* label) {
    for (float value : values) {
      if (!std::isfinite(value)) throw std::runtime_error(std::string("SuperMimic nonfinite ") + label);
    }
  }

  static void require_joint_values(const std::vector<float>& values, const char* label) {
    if (values.size() != kActionDim) {
      throw std::runtime_error(std::string("SuperMimic ") + label + " must have 29 values");
    }
    require_finite(values, label);
  }

  static Rotation identity_rotation() { return {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}; }

  static Rotation transpose(const Rotation& matrix) {
    return {matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8]};
  }

  static Rotation multiply(const Rotation& left, const Rotation& right) {
    Rotation result{};
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t column = 0; column < 3; ++column) {
        for (std::size_t inner = 0; inner < 3; ++inner) {
          result[3 * row + column] += left[3 * row + inner] * right[3 * inner + column];
        }
      }
    }
    return result;
  }

  bool configured_{false};
  bool history_initialized_{false};
  uint64_t policy_step_{0};
  std::size_t frame_limit_{0};
  std::array<float, kActionDim> default_q_{};
  std::array<float, kActionDim> action_scale_{};
  std::array<float, kActionDim> previous_action_{};
  std::array<Observation, kHistoryLen> history_{};
  Rotation initial_to_world_{identity_rotation()};
  std::vector<SuperMimicReferenceFrame> reference_;
};

}
