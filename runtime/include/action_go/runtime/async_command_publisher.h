#pragma once

#include "action_go/runtime/activation_blend.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace action_go::runtime {

struct AsyncPublisherMetrics {
  int64_t publish_count{0};
  int64_t deadline_miss{0};
  double max_publish_interval_ms{0.0};
};

class AsyncCommandPublisher {
 public:
  using Clock = std::chrono::steady_clock;
  using PublishFn = std::function<void(const JointCommand&, bool active)>;

  AsyncCommandPublisher(int publish_hz, double target_interp_sec,
                        JointCommand initial_hold, PublishFn publish_fn,
                        bool initially_active = false);
  ~AsyncCommandPublisher();

  AsyncCommandPublisher(const AsyncCommandPublisher&) = delete;
  AsyncCommandPublisher& operator=(const AsyncCommandPublisher&) = delete;

  void update(JointCommand command, bool active = true,
              Clock::time_point now = Clock::now());
  void stop_and_join();
  bool failed() const;
  void rethrow_if_failed() const;
  AsyncPublisherMetrics metrics() const;

 private:
  static JointCommand interpolate(const JointCommand& from,
                                  const JointCommand& to, double alpha);
  void run();

  const std::chrono::steady_clock::duration publish_period_;
  const double target_interp_sec_;
  PublishFn publish_fn_;

  mutable std::mutex command_mutex_;
  JointCommand begin_;
  JointCommand goal_;
  Clock::time_point update_time_{};
  bool active_{false};

  std::atomic<bool> stop_{false};
  std::atomic<bool> failed_{false};
  mutable std::mutex error_mutex_;
  std::exception_ptr error_;
  std::thread thread_;

  std::atomic<int64_t> publish_count_{0};
  std::atomic<int64_t> deadline_miss_{0};
  std::atomic<double> max_publish_interval_ms_{0.0};
};

}  // namespace action_go::runtime
