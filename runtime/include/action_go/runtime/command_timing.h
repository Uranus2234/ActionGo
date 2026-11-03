#pragma once

#include "action_go/runtime/event_logger.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace action_go::runtime {

struct CommandTiming {
  std::chrono::steady_clock::time_point received{}, inference_started{}, inference_finished{};
  uint64_t state_sequence{0}, command_sequence{0};
};

inline thread_local CommandTiming inference_timing;

inline void begin_command_timing(std::chrono::steady_clock::time_point received,
                                 uint64_t state_sequence) {
  static std::atomic<uint64_t> sequence{0};
  inference_timing = {received, std::chrono::steady_clock::now(), {},
                      state_sequence, ++sequence};
}

inline CommandTiming finish_command_timing() {
  auto timing = inference_timing;
  if (timing.command_sequence && timing.inference_finished.time_since_epoch().count() == 0)
    timing.inference_finished = std::chrono::steady_clock::now();
  return timing;
}

class CommandTimingScope {
 public:
  CommandTimingScope(std::chrono::steady_clock::time_point received, uint64_t sequence) {
    begin_command_timing(received, sequence);
  }
  ~CommandTimingScope() { inference_timing.inference_finished = std::chrono::steady_clock::now(); }
};

class CommandTimingReporter {
 public:
  void published(RuntimeEventLogger& logger, const CommandTiming& timing) {
    if (!timing.command_sequence || timing.received.time_since_epoch().count() == 0) return;
    const auto published = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    if (timing.command_sequence == last_sequence_) return;
    last_sequence_ = timing.command_sequence;
    if (published - last_record_ < std::chrono::milliseconds(200)) return;
    last_record_ = published;
    const auto milliseconds = [](auto duration) {
      return std::chrono::duration<double, std::milli>(duration).count();
    };
    if (timing.inference_finished < timing.inference_started ||
        timing.inference_started < timing.received) return;
    logger.log("command_timing",
               {event_int("state_seq", timing.state_sequence),
                event_int("command_seq", timing.command_sequence),
                event_number("state_wait_ms", milliseconds(timing.inference_started - timing.received)),
                event_number("inference_ms", milliseconds(timing.inference_finished - timing.inference_started)),
                event_number("publish_wait_ms", milliseconds(published - timing.inference_finished)),
                event_number("e2e_ms", milliseconds(published - timing.received))});
  }
 private:
  std::mutex mutex_;
  uint64_t last_sequence_{0};
  std::chrono::steady_clock::time_point last_record_{};
};

}
