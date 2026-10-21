#include "action_go/runtime/bfm_observation.h"

#include <algorithm>
#include <stdexcept>

namespace action_go::runtime {

std::vector<float> BfmObservation::actor_input(const std::vector<float>& latent) const {
  std::vector<float> result;
  result.reserve(state.size() + last_action.size() + history.size() + latent.size());
  for (const auto* values : {&state, &last_action, &history, &latent}) {
    result.insert(result.end(), values->begin(), values->end());
  }
  return result;
}

void BfmObservationHistory::reset(size_t motor_count, size_t history_steps) {
  motor_count_ = motor_count;
  history_steps_ = history_steps;
  const std::array<size_t, 5> dimensions{motor_count, 3, motor_count, motor_count, 3};
  for (size_t field = 0; field < histories_.size(); ++field) {
    histories_[field].assign(dimensions[field] * history_steps, 0.f);
  }
}

BfmObservation BfmObservationHistory::build(
    const std::vector<float>& position, const std::vector<float>& velocity,
    const std::vector<float>& default_position, const std::array<float, 3>& gravity,
    const std::array<float, 3>& angular_velocity, const std::vector<float>& scaled_action,
    BfmHistoryTiming timing, float angular_velocity_scale) {
  if (motor_count_ == 0 || position.size() < motor_count_ || velocity.size() < motor_count_ ||
      default_position.size() != motor_count_ || scaled_action.size() != motor_count_) {
    throw std::runtime_error("BFM observation motor dimension mismatch");
  }
  std::vector<float> relative_position(motor_count_), joint_velocity(motor_count_), gyro(3);
  for (size_t joint = 0; joint < motor_count_; ++joint) {
    relative_position[joint] = position[joint] - default_position[joint];
    joint_velocity[joint] = velocity[joint];
  }
  for (size_t axis = 0; axis < gyro.size(); ++axis) {
    gyro[axis] = angular_velocity[axis] * angular_velocity_scale;
  }
  const std::array<std::vector<float>, 5> fields{
      scaled_action, gyro, relative_position, joint_velocity,
      std::vector<float>(gravity.begin(), gravity.end())};
  auto advance = [&]() {
    if (history_steps_ == 0) return;
    for (size_t field = 0; field < histories_.size(); ++field) {
      auto& history = histories_[field];
      const auto& value = fields[field];
      std::move_backward(history.begin(), history.end() - value.size(), history.end());
      std::copy(value.begin(), value.end(), history.begin());
    }
  };
  if (timing == BfmHistoryTiming::kIncludeCurrent) advance();
  BfmObservation observation;
  observation.state = std::move(relative_position);
  observation.state.insert(observation.state.end(), joint_velocity.begin(), joint_velocity.end());
  observation.state.insert(observation.state.end(), gravity.begin(), gravity.end());
  observation.state.insert(observation.state.end(), gyro.begin(), gyro.end());
  observation.last_action = scaled_action;
  for (const auto& history : histories_) {
    observation.history.insert(observation.history.end(), history.begin(), history.end());
  }
  if (timing == BfmHistoryTiming::kPreviousOnly) advance();
  return observation;
}

}
