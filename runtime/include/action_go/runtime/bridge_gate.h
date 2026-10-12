#pragma once

#include <algorithm>
#include <cstdint>

namespace action_go::runtime {

enum class BridgeCommand { kNone, kStartBfm, kSwitchToB, kDamping };

enum class BridgeState { kIdle, kBfmStand, kEmergencyObserve, kDamped };

struct BridgeGateConfig {
  float level1_max_percent{8.01123025F};
  float level2_max_percent{10.5937071F};
  float level3_max_percent{40.0F};
  int normal_confirm_steps{2};
  int stand_confirm_steps{5};
  int level3_observe_steps{50};
  int level4_observe_steps{25};
};

struct BridgeGateOutput {
  BridgeCommand command{BridgeCommand::kNone};
  BridgeState state{BridgeState::kIdle};
  int risk_level{1};
  int confirmation_steps{0};
  int observation_steps{0};
  bool rejected{false};
};

class BridgeGate {
 public:
  explicit BridgeGate(BridgeGateConfig config = {}) : config_(config) {}

  void reset() {
    state_ = BridgeState::kIdle;
    normal_steps_ = 0;
    stand_steps_ = 0;
    observation_steps_ = 0;
    observation_start_level_ = 0;
  }

  BridgeGateOutput tick(float risk_percent, bool normal_request, bool emergency_request,
                        bool stand_ready) {
    BridgeGateOutput out;
    out.state = state_;
    out.risk_level = level(risk_percent);
    if (state_ == BridgeState::kDamped) return out;

    if (state_ == BridgeState::kBfmStand) {
      if (stand_ready) ++stand_steps_;
      else stand_steps_ = 0;
      out.confirmation_steps = stand_steps_;
      if (stand_steps_ >= std::max(1, config_.stand_confirm_steps)) {
        state_ = BridgeState::kIdle;
        stand_steps_ = 0;
        out.command = BridgeCommand::kSwitchToB;
        out.state = state_;
      }
      return out;
    }

    if (state_ == BridgeState::kEmergencyObserve) {
      ++observation_steps_;
      out.observation_steps = observation_steps_;
      const int current_level = level(risk_percent);
      if (current_level < observation_start_level_) {
        state_ = BridgeState::kBfmStand;
        stand_steps_ = 0;
        out.command = BridgeCommand::kStartBfm;
        out.state = state_;
        return out;
      }
      const int limit = observation_start_level_ >= 4 ? config_.level4_observe_steps
                                                       : config_.level3_observe_steps;
      if (observation_steps_ >= std::max(1, limit)) {
        state_ = BridgeState::kDamped;
        out.command = BridgeCommand::kDamping;
        out.state = state_;
      }
      return out;
    }

    if (emergency_request) {
      if (out.risk_level <= 2) {
        ++normal_steps_;
        out.confirmation_steps = normal_steps_;
        if (normal_steps_ >= std::max(1, config_.normal_confirm_steps)) {
          state_ = BridgeState::kBfmStand;
          normal_steps_ = 0;
          out.command = BridgeCommand::kStartBfm;
          out.state = state_;
        }
      } else if (out.risk_level >= 3) {
        state_ = BridgeState::kEmergencyObserve;
        observation_start_level_ = out.risk_level;
        observation_steps_ = 0;
        normal_steps_ = 0;
        out.state = state_;
      }
      return out;
    }

    if (normal_request) {
      if (out.risk_level <= 2) {
        ++normal_steps_;
        out.confirmation_steps = normal_steps_;
        if (normal_steps_ >= std::max(1, config_.normal_confirm_steps)) {
          state_ = BridgeState::kBfmStand;
          normal_steps_ = 0;
          out.command = BridgeCommand::kStartBfm;
          out.state = state_;
        }
      } else {
        normal_steps_ = 0;
        out.rejected = true;
      }
    } else {
      normal_steps_ = 0;
    }
    return out;
  }

  BridgeState state() const { return state_; }

  int level(float risk_percent) const {
    if (!(risk_percent >= 0.0F)) return 4;
    if (risk_percent < config_.level1_max_percent) return 1;
    if (risk_percent < config_.level2_max_percent) return 2;
    if (risk_percent < config_.level3_max_percent) return 3;
    return 4;
  }

 private:
  BridgeGateConfig config_;
  BridgeState state_{BridgeState::kIdle};
  int normal_steps_{0};
  int stand_steps_{0};
  int observation_steps_{0};
  int observation_start_level_{0};
};

class DoubleClickCombo {
 public:
  struct Config { int hold_steps{25}; int double_click_window_steps{20}; };
  DoubleClickCombo() = default;
  explicit DoubleClickCombo(Config config) : config_(config) {}

  bool update(bool modifier_down, bool button_down) {
    ++step_;
    if (!modifier_down) { hold_steps_ = 0; clicks_ = 0; waiting_release_ = false; return false; }
    if (button_down && !previous_button_down_) {
      if (waiting_release_ && step_ - last_click_step_ <= config_.double_click_window_steps) {
        waiting_release_ = false;
        clicks_ = 0;
        previous_button_down_ = true;
        return hold_steps_ >= config_.hold_steps;
      }
      waiting_release_ = true;
      last_click_step_ = step_;
      ++clicks_;
    }
    if (!button_down && previous_button_down_ && clicks_ > 0 &&
        step_ - last_click_step_ > config_.double_click_window_steps) {
      waiting_release_ = false;
      clicks_ = 0;
    }
    previous_button_down_ = button_down;
    ++hold_steps_;
    return false;
  }

 private:
  Config config_;
  int step_{0};
  int hold_steps_{0};
  int clicks_{0};
  int last_click_step_{0};
  bool previous_button_down_{false};
  bool waiting_release_{false};
};

}  // namespace action_go::runtime
