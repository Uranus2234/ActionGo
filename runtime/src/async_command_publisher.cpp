#include "action_go/runtime/async_command_publisher.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace action_go::runtime {
namespace {

void validate_command(const JointCommand& command) {
  if (command.q.empty() || command.q.size() != command.kp.size() ||
      command.q.size() != command.kd.size()) {
    throw std::invalid_argument("async publisher q/kp/kd dimensions must match");
  }
}

}  // namespace

AsyncCommandPublisher::AsyncCommandPublisher(int publish_hz,
                                             double target_interp_sec,
                                             JointCommand initial_hold,
                                             PublishFn publish_fn,
                                             bool initially_active)
    : publish_period_(std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(1.0 / std::max(1, publish_hz)))),
      target_interp_sec_(target_interp_sec),
      publish_fn_(std::move(publish_fn)),
      begin_(std::move(initial_hold)),
      goal_(begin_),
      update_time_(Clock::now()),
      active_(initially_active) {
  validate_command(begin_);
  if (!std::isfinite(target_interp_sec_) || target_interp_sec_ < 0.0) {
    throw std::invalid_argument("target_interp_sec must be finite and non-negative");
  }
  if (!publish_fn_) throw std::invalid_argument("async publisher callback is required");
  thread_ = std::thread(&AsyncCommandPublisher::run, this);
}

AsyncCommandPublisher::~AsyncCommandPublisher() {
  stop_and_join();
}

JointCommand AsyncCommandPublisher::interpolate(const JointCommand& from,
                                                const JointCommand& to,
                                                double alpha) {
  validate_command(from);
  validate_command(to);
  if (from.q.size() != to.q.size()) {
    throw std::invalid_argument("async publisher command dimensions changed");
  }
  const float a = static_cast<float>(std::clamp(alpha, 0.0, 1.0));
  JointCommand out = to;
  for (size_t i = 0; i < out.q.size(); ++i) {
    out.q[i] = from.q[i] + (to.q[i] - from.q[i]) * a;
  }
  return out;
}

void AsyncCommandPublisher::update(JointCommand command, bool active,
                                   Clock::time_point now) {
  validate_command(command);
  std::lock_guard<std::mutex> lock(command_mutex_);
  const double alpha = target_interp_sec_ > 1e-6
      ? std::chrono::duration<double>(now - update_time_).count() /
            target_interp_sec_
      : 1.0;
  begin_ = interpolate(begin_, goal_, alpha);
  goal_ = std::move(command);
  update_time_ = now;
  active_ = active;
}

void AsyncCommandPublisher::stop_and_join() {
  stop_.store(true);
  if (thread_.joinable()) thread_.join();
}

bool AsyncCommandPublisher::failed() const {
  return failed_.load();
}

void AsyncCommandPublisher::rethrow_if_failed() const {
  std::lock_guard<std::mutex> lock(error_mutex_);
  if (error_) std::rethrow_exception(error_);
}

AsyncPublisherMetrics AsyncCommandPublisher::metrics() const {
  return {publish_count_.load(), deadline_miss_.load(),
          max_publish_interval_ms_.load()};
}

void AsyncCommandPublisher::run() {
  try {
    auto next_publish = Clock::now();
    auto previous_publish = Clock::time_point{};
    while (!stop_.load()) {
      const auto now = Clock::now();
      JointCommand command;
      bool active = false;
      {
        std::lock_guard<std::mutex> lock(command_mutex_);
        const double alpha = target_interp_sec_ > 1e-6
            ? std::chrono::duration<double>(now - update_time_).count() /
                  target_interp_sec_
            : 1.0;
        command = interpolate(begin_, goal_, alpha);
        active = active_;
      }
      publish_fn_(command, active);
      const auto published_at = Clock::now();
      if (previous_publish.time_since_epoch().count() != 0) {
        const double interval_ms =
            std::chrono::duration<double, std::milli>(published_at - previous_publish).count();
        double observed = max_publish_interval_ms_.load();
        while (interval_ms > observed &&
               !max_publish_interval_ms_.compare_exchange_weak(observed, interval_ms)) {
        }
      }
      previous_publish = published_at;
      publish_count_.fetch_add(1);

      next_publish += publish_period_;
      const auto after_publish = Clock::now();
      if (next_publish < after_publish) {
        do {
          next_publish += publish_period_;
          deadline_miss_.fetch_add(1);
        } while (next_publish < after_publish);
      }
      if (next_publish > Clock::now()) std::this_thread::sleep_until(next_publish);
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_ = std::current_exception();
    }
    failed_.store(true);
    stop_.store(true);
  }
}

}  // namespace action_go::runtime
