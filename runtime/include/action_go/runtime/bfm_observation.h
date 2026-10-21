#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace action_go::runtime {

enum class BfmHistoryTiming { kIncludeCurrent, kPreviousOnly };

struct BfmObservation {
  std::vector<float> state;
  std::vector<float> last_action;
  std::vector<float> history;

  std::vector<float> actor_input(const std::vector<float>& latent) const;
};

class BfmObservationHistory {
 public:
  void reset(size_t motor_count, size_t history_steps);
  BfmObservation build(const std::vector<float>& position,
                       const std::vector<float>& velocity,
                       const std::vector<float>& default_position,
                       const std::array<float, 3>& gravity,
                       const std::array<float, 3>& angular_velocity,
                       const std::vector<float>& scaled_action,
                       BfmHistoryTiming timing,
                       float angular_velocity_scale = 0.25f);

 private:
  size_t motor_count_{0};
  size_t history_steps_{0};
  std::array<std::vector<float>, 5> histories_;
};

}
