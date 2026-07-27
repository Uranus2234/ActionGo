#include "action_go/runtime/input_freshness.h"

#include <algorithm>

namespace action_go::runtime {

void RealtimeInputStaleGate::reset() {
  stale_active_ = false;
  stale_start_ = {};
}

StaleGateDecision RealtimeInputStaleGate::decide(SampleFreshness freshness,
                                                 StaleBehavior behavior,
                                                 double stop_timeout_sec,
                                                 std::chrono::steady_clock::time_point now) {
  if (freshness == SampleFreshness::kFresh) {
    reset();
    return {StaleDecisionAction::kUseSample, 0.0, false};
  }

  if (behavior == StaleBehavior::kDefault) {
    reset();
    return {StaleDecisionAction::kUseDefault, 0.0, false};
  }
  if (behavior == StaleBehavior::kStop) {
    reset();
    return {StaleDecisionAction::kStop, 0.0, true};
  }

  StaleGateDecision decision;
  decision.action = StaleDecisionAction::kHoldLast;
  if (!stale_active_) {
    stale_active_ = true;
    stale_start_ = now;
    decision.new_stale_event = true;
  }
  decision.hold_elapsed_sec = std::chrono::duration<double>(now - stale_start_).count();
  if (decision.hold_elapsed_sec >= std::max(0.0, stop_timeout_sec)) {
    decision.action = StaleDecisionAction::kStop;
  }
  return decision;
}

}  // namespace action_go::runtime
