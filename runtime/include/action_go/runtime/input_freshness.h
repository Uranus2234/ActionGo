#pragma once

#include <chrono>

namespace action_go::runtime {

enum class SampleFreshness {
  kMissing,
  kStale,
  kFresh,
};

enum class StaleBehavior {
  kDefault,
  kHoldThenStop,
  kStop,
};

enum class StaleDecisionAction {
  kUseSample,
  kUseDefault,
  kHoldLast,
  kStop,
};

struct StaleGateDecision {
  StaleDecisionAction action{StaleDecisionAction::kStop};
  double hold_elapsed_sec{0.0};
  bool new_stale_event{false};
};

class RealtimeInputStaleGate {
 public:
  void reset();

  StaleGateDecision decide(SampleFreshness freshness,
                           StaleBehavior behavior,
                           double stop_timeout_sec,
                           std::chrono::steady_clock::time_point now);

 private:
  bool stale_active_{false};
  std::chrono::steady_clock::time_point stale_start_{};
};

}  // namespace action_go::runtime
