#pragma once

#include "action_go/runtime/supermimic_policy.h"

namespace action_go::runtime {

class AllMimicController {
 public:
  static constexpr std::size_t kActionDim = 29;
  static constexpr std::size_t kObsFrameDim = 93;
  static constexpr std::size_t kHistoryLen = 10;
  static constexpr std::size_t kGoalDim = 730;
  static constexpr std::size_t kObsDim = 1660;
  static constexpr uint32_t kPolicyTickMs = 20;
  using Rotation = SuperMimicController::Rotation;
  using Observation = std::array<float, kObsFrameDim>;

  void configure(const std::vector<float>& defaults, const std::vector<float>& scales,
                 const std::vector<float>& residual_scale, const std::vector<float>& residual_reference) {
    require_joints(defaults);
    require_joints(scales);
    require_joints(residual_scale);
    require_joints(residual_reference);
    defaults_ = defaults;
    scales_ = scales;
    residual_scale_ = residual_scale;
    residual_reference_ = residual_reference;
    reference_.configure(defaults, scales);
  }

  void load_reference_csv(const std::string& path) { reference_.load_reference_csv(path); }
  void set_reference_frames(std::vector<SuperMimicReferenceFrame> frames) {
    reference_.set_reference_frames(std::move(frames));
  }
  void set_frame_limit(std::size_t count) { reference_.set_frame_limit(count); }
  std::size_t reference_size() const { return reference_.reference_size(); }
  std::size_t frame_limit() const { return reference_.frame_limit(); }
  std::size_t reference_index() const { return reference_.reference_index(); }
  uint64_t policy_step() const { return reference_.policy_step(); }
  bool at_end() const { return reference_.at_end(); }

  void reset(const std::array<float, 4>& quat, bool align_initial_yaw) {
    reference_.reset(quat, align_initial_yaw);
    initialized_ = false;
    goal_ready_ = false;
  }

  std::vector<float> build_input(const SuperMimicRobotFrame& robot) {
    const auto shared = reference_.build_input(robot);
    Observation obs{};
    std::copy_n(shared.end() - 730 - 99, 93, obs.begin());
    if (!initialized_) {
      history_.fill(obs);
      initialized_ = true;
    } else {
      for (std::size_t index = 1; index < kHistoryLen; ++index) history_[index - 1] = history_[index];
      history_.back() = obs;
    }
    std::vector<float> packed;
    packed.reserve(kObsDim);
    for (const auto& frame : history_) packed.insert(packed.end(), frame.begin(), frame.end());
    goal_.assign(shared.end() - kGoalDim, shared.end());
    const auto rotation = SuperMimicController::rotation_from_quaternion(robot.quat);
    const auto& alignment = reference_.initial_to_world();
    for (std::size_t frame = 0; frame < 10; ++frame) {
      const std::size_t offset = frame * 73 + 3;
      std::array<float, 6> relative{};
      for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
          for (std::size_t world_axis = 0; world_axis < 3; ++world_axis) {
            for (std::size_t ref_axis = 0; ref_axis < 3; ++ref_axis) {
              relative[3 * column + row] += rotation[3 * world_axis + row] *
                  alignment[3 * world_axis + ref_axis] * goal_[offset + 3 * column + ref_axis];
            }
          }
        }
      }
      std::copy(relative.begin(), relative.end(), goal_.begin() + offset);
    }
    packed.insert(packed.end(), goal_.begin(), goal_.end());
    goal_ready_ = true;
    return packed;
  }

  std::vector<float> decode_action(const std::vector<float>& action) const {
    require_joints(action);
    if (!goal_ready_) throw std::runtime_error("AllMimic requires build_input before action decoding");
    std::vector<float> target(kActionDim);
    for (std::size_t joint = 0; joint < kActionDim; ++joint) {
      target[joint] = defaults_[joint] + scales_[joint] * residual_scale_[joint] * action[joint] +
          residual_reference_[joint] * (goal_[9 + joint] - defaults_[joint]);
    }
    for (float value : target) {
      if (!std::isfinite(value)) throw std::runtime_error("AllMimic nonfinite joint target");
    }
    return target;
  }

  void accept_action(const std::vector<float>& action) {
    require_joints(action);
    reference_.accept_action(action);
  }

 private:
  static void require_joints(const std::vector<float>& values) {
    if (values.size() != kActionDim) throw std::runtime_error("AllMimic requires 29 joint values");
    for (float value : values) {
      if (!std::isfinite(value)) throw std::runtime_error("AllMimic nonfinite joint value");
    }
  }

  SuperMimicController reference_;
  bool initialized_{false};
  bool goal_ready_{false};
  std::array<Observation, kHistoryLen> history_{};
  std::vector<float> defaults_, scales_, residual_scale_, residual_reference_, goal_;
};

}
