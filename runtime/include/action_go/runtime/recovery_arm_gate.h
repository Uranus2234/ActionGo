#pragma once

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace action_go::runtime {

class RecoveryArmGate {
 public:
  using Clock = std::chrono::steady_clock;

  struct Config {
    double fall_angle_deg{57.3};
    std::chrono::milliseconds fall_hold{60};
    std::chrono::milliseconds imu_stale_after{500};
  };

  explicit RecoveryArmGate(Config config) : config_(config) {
    if (!std::isfinite(config_.fall_angle_deg) || config_.fall_angle_deg <= 0.0 ||
        config_.fall_angle_deg > 180.0 || config_.fall_hold.count() <= 0 ||
        config_.imu_stale_after.count() <= 0) {
      throw std::invalid_argument("invalid recovery ARM gate configuration");
    }
  }

  void arm(std::string recovery_backend) {
    if (recovery_backend.empty()) {
      throw std::invalid_argument("recovery backend must not be empty");
    }
    recovery_backend_ = std::move(recovery_backend);
    armed_ = true;
    triggered_ = false;
    timer_active_ = false;
  }

  void disarm() {
    armed_ = false;
    triggered_ = false;
    timer_active_ = false;
  }

  void reset_debounce() {
    triggered_ = false;
    timer_active_ = false;
  }

  bool observe(bool imu_valid, double tilt_deg, Clock::time_point imu_stamp,
               Clock::time_point now = Clock::now()) {
    if (!armed_ || triggered_) return false;
    if (!imu_valid || !std::isfinite(tilt_deg) ||
        imu_stamp.time_since_epoch().count() == 0 ||
        now < imu_stamp || now - imu_stamp > config_.imu_stale_after) {
      timer_active_ = false;
      return false;
    }
    if (tilt_deg < config_.fall_angle_deg) {
      timer_active_ = false;
      return false;
    }
    if (!timer_active_) {
      fall_since_ = now;
      timer_active_ = true;
      return false;
    }
    if (now - fall_since_ < config_.fall_hold) return false;
    triggered_ = true;
    timer_active_ = false;
    return true;
  }

  bool armed() const { return armed_; }
  bool triggered() const { return triggered_; }
  const std::string& recovery_backend() const { return recovery_backend_; }

 private:
  Config config_;
  std::string recovery_backend_;
  bool armed_{false};
  bool triggered_{false};
  bool timer_active_{false};
  Clock::time_point fall_since_{};
};

}  // namespace action_go::runtime
