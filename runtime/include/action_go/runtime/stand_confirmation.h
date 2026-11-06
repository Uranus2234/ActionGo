#pragma once

#include <chrono>
#include <cmath>

namespace action_go::runtime {

class StandConfirmation {
 public:
  using Clock = std::chrono::steady_clock;

  void reset() { steps_ = 0; last_stamp_ = {}; }

  bool update(bool valid, double tilt_rad, double gyro_norm,
              Clock::time_point stamp, Clock::time_point now) {
    if (!valid || !std::isfinite(tilt_rad) || !std::isfinite(gyro_norm) ||
        tilt_rad < 0.0 || gyro_norm < 0.0 || tilt_rad >= 0.20 || gyro_norm >= 0.30 ||
        stamp == Clock::time_point{} || stamp > now || now - stamp >= std::chrono::milliseconds(100)) {
      reset();
      return false;
    }
    if (stamp > last_stamp_) {
      last_stamp_ = stamp;
      if (steps_ < 5) ++steps_;
    }
    return steps_ >= 5;
  }

 private:
  int steps_{0};
  Clock::time_point last_stamp_{};
};

}
