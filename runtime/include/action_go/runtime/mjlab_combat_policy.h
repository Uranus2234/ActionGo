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
    if (align_initial_yaw) throw std::runtime_error("MJLab combat requires yaw_alignment=none");
    decoder_.reset(quat, false);
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
    const auto& current = reference_[frame];
    const auto& future = reference_[std::min(frame + 2, reference_.size() - 1)];
    input.insert(input.end(), current.begin(), current.end());
    input.insert(input.end(), future.begin(), future.end());
    input.push_back(frame + 2 < reference_.size() ? 1.f : 0.f);
    for (float value : input) {
      if (!std::isfinite(value)) throw std::runtime_error("MJLab combat nonfinite raw input");
    }
    return input;
  }

  std::vector<float> decode_action(const std::vector<float>& action) const { return decoder_.decode_action(action); }
  void accept_action(const std::vector<float>& action) { decoder_.accept_action(action); }

 private:
  SuperMimicController decoder_;
  std::vector<Reference> reference_;
  std::size_t frame_limit_{0};
};

}
