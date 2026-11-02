#pragma once

#include "action_go/runtime/supermimic_policy.h"

namespace action_go::runtime {

class MjlabCombatController {
 public:
  static constexpr std::size_t kReferenceDim = 83;
  static constexpr std::size_t kObsDim = 261;
  using Reference = std::array<float, kReferenceDim>;

  void configure(const std::vector<float>& defaults, const std::vector<float>& scales) {
    decoder_.configure(defaults, scales);
  }

  void load_reference_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("MJLab combat reference not found: " + path);
    std::vector<Reference> frames;
    std::string line;
    while (std::getline(input, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      Reference frame{};
      std::size_t begin = 0;
      for (std::size_t index = 0; index < frame.size(); ++index) {
        const auto end = line.find(',', begin);
        if ((index + 1 < frame.size()) != (end != std::string::npos)) {
          throw std::runtime_error("MJLab combat reference requires exactly 83 columns");
        }
        std::istringstream token(line.substr(begin, end == std::string::npos ? end : end - begin));
        token.imbue(std::locale::classic());
        if (!(token >> frame[index]) || !std::isfinite(frame[index])) {
          throw std::runtime_error("MJLab combat reference has an invalid/nonfinite value");
        }
        token >> std::ws;
        if (!token.eof()) throw std::runtime_error("MJLab combat reference has a malformed float");
        if (end != std::string::npos) begin = end + 1;
      }
      float norm = 0.f;
      for (std::size_t index = 61; index < 65; ++index) norm += frame[index] * frame[index];
      if (std::abs(norm - 1.f) > 1e-3f) {
        throw std::runtime_error("MJLab combat anchor quaternion must be normalized wxyz");
      }
      frames.push_back(frame);
    }
    if (input.bad() || frames.size() < 2) {
      throw std::runtime_error("MJLab combat reference must contain at least two valid frames");
    }
    reference_ = std::move(frames);
    frame_limit_ = reference_.size();
    reset({1.f, 0.f, 0.f, 0.f}, false);
  }

  void set_frame_limit(std::size_t count) {
    if (count == 0 || count > reference_.size()) {
      throw std::runtime_error("MJLab combat frame limit exceeds reference length");
    }
    frame_limit_ = count;
  }

  std::size_t reference_size() const { return reference_.size(); }
  std::size_t frame_limit() const { return frame_limit_; }
  uint64_t policy_step() const { return decoder_.policy_step(); }
  bool at_end() const { return policy_step() >= frame_limit_; }
  std::size_t reference_index() const {
    if (!frame_limit_) throw std::runtime_error("MJLab combat reference is not loaded");
    return static_cast<std::size_t>(std::min<uint64_t>(policy_step(), frame_limit_ - 1));
  }

  void reset(const std::array<float, 4>& quat, bool align_initial_yaw) {
    SuperMimicRobotFrame robot;
    robot.quat = quat;
    reset_from_state(robot, align_initial_yaw);
  }

  void reset_from_state(const SuperMimicRobotFrame& robot, bool align_initial_yaw) {
    decoder_.reset(robot.quat, false);
    yaw_offset_ = 0.f;
    if (align_initial_yaw) {
      if (reference_.empty()) throw std::runtime_error("MJLab combat reference is not loaded");
      for (float position : robot.q) {
        if (!std::isfinite(position)) throw std::runtime_error("MJLab combat nonfinite reset joint state");
      }
      const auto waist_yaw = axis_quaternion(robot.q[12], 2);
      const auto waist_roll = axis_quaternion(robot.q[13], 0);
      const auto waist_pitch = axis_quaternion(robot.q[14], 1);
      const auto torso = multiply_quaternion(robot.quat,
          multiply_quaternion(waist_yaw, multiply_quaternion(waist_roll, waist_pitch)));
      const auto robot_rotation = SuperMimicController::rotation_from_quaternion(torso);
      const auto& first = reference_.front();
      const auto reference_rotation = SuperMimicController::rotation_from_quaternion(
          {first[61], first[62], first[63], first[64]});
      yaw_offset_ = std::atan2(robot_rotation[3], robot_rotation[0]) -
          std::atan2(reference_rotation[3], reference_rotation[0]);
      yaw_offset_ = std::atan2(std::sin(yaw_offset_), std::cos(yaw_offset_));
    }
  }

  float yaw_offset() const { return yaw_offset_; }

  Reference aligned_reference(std::size_t frame) const {
    Reference result = reference_.at(frame);
    if (yaw_offset_ == 0.f) return result;
    const float cosine = std::cos(yaw_offset_), sine = std::sin(yaw_offset_);
    const auto rotate = [&](std::size_t offset, bool position) {
      const float origin_x = position ? reference_.front()[58] : 0.f;
      const float origin_y = position ? reference_.front()[59] : 0.f;
      const float horizontal_x = result[offset] - origin_x;
      const float horizontal_y = result[offset + 1] - origin_y;
      result[offset] = origin_x + cosine * horizontal_x - sine * horizontal_y;
      result[offset + 1] = origin_y + sine * horizontal_x + cosine * horizontal_y;
    };
    rotate(58, true);
    rotate(65, false);
    rotate(68, false);
    for (std::size_t offset = 71; offset < kReferenceDim; offset += 3) rotate(offset, true);
    const auto orientation = multiply_quaternion(axis_quaternion(yaw_offset_, 2),
        {result[61], result[62], result[63], result[64]});
    std::copy(orientation.begin(), orientation.end(), result.begin() + 61);
    return result;
  }

  std::vector<float> build_input(const SuperMimicRobotFrame& robot) {
    SuperMimicController::rotation_from_quaternion(robot.quat);
    std::vector<float> input;
    input.reserve(kObsDim);
    input.insert(input.end(), robot.q.begin(), robot.q.end());
    input.insert(input.end(), robot.dq.begin(), robot.dq.end());
    input.insert(input.end(), robot.quat.begin(), robot.quat.end());
    input.insert(input.end(), robot.gyro.begin(), robot.gyro.end());
    const auto& previous = decoder_.previous_action();
    input.insert(input.end(), previous.begin(), previous.end());
    const auto frame = reference_index();
    const auto current = aligned_reference(frame);
    const auto future = aligned_reference(std::min(frame + 2, frame_limit_ - 1));
    input.insert(input.end(), current.begin(), current.end());
    input.insert(input.end(), future.begin(), future.end());
    input.push_back(frame + 2 < frame_limit_ ? 1.f : 0.f);
    for (float value : input) {
      if (!std::isfinite(value)) throw std::runtime_error("MJLab combat nonfinite raw input");
    }
    return input;
  }

  std::vector<float> decode_action(const std::vector<float>& action) const { return decoder_.decode_action(action); }
  void accept_action(const std::vector<float>& action) { decoder_.accept_action(action); }

 private:
  static std::array<float, 4> axis_quaternion(float angle, std::size_t axis) {
    std::array<float, 4> result{std::cos(angle * 0.5f), 0.f, 0.f, 0.f};
    result[axis + 1] = std::sin(angle * 0.5f);
    return result;
  }

  static std::array<float, 4> multiply_quaternion(const std::array<float, 4>& left,
                                                const std::array<float, 4>& right) {
    return {left[0] * right[0] - left[1] * right[1] - left[2] * right[2] - left[3] * right[3],
            left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2],
            left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
            left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0]};
  }

  SuperMimicController decoder_;
  std::vector<Reference> reference_;
  std::size_t frame_limit_{0};
  float yaw_offset_{0.f};
};

}
