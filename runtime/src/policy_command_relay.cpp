#include "action_go/runtime/policy_command_relay.h"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace action_go::runtime {
namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using Clock = std::chrono::steady_clock;

class DdsPolicyCommandRelay final : public PolicyCommandRelay {
 public:
  DdsPolicyCommandRelay(const std::string& output_topic, const std::string& state_topic)
      : output_topic_(output_topic), publisher_(output_topic), state_(state_topic) {
    publisher_.InitChannel();
    state_.InitChannel([this](const void* message) {
      const auto& state = *static_cast<const LowState*>(message);
      bool valid = true;
      for (const auto& motor : state.motor_state()) {
        valid = valid && std::isfinite(motor.q()) && std::isfinite(motor.dq());
      }
      for (float value : state.imu_state().quaternion()) valid = valid && std::isfinite(value);
      for (float value : state.imu_state().gyroscope()) valid = valid && std::isfinite(value);
      std::lock_guard<std::mutex> lock(mutex_);
      state_stamp_ = valid ? Clock::now() : Clock::time_point{};
    }, 1);
    worker_ = std::thread([this]() {
      auto next = Clock::now();
      while (!stop_.load()) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          const auto selected = inputs_.find(selected_);
          if (selected != inputs_.end() && fresh(*selected->second)) {
            publisher_.Write(selected->second->command);
          }
        }
        next += std::chrono::milliseconds(2);
        if (next < Clock::now()) next = Clock::now();
        std::this_thread::sleep_until(next);
      }
    });
  }

  ~DdsPolicyCommandRelay() override {
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
    state_.CloseChannel();
    for (auto& input : inputs_) input.second->subscriber->CloseChannel();
  }

  void add(const std::string& topic) override {
    if (topic.empty() || topic == output_topic_ || topic == "rt/lowcmd") {
      throw std::invalid_argument("policy relay requires a private command topic");
    }
    auto input = std::make_shared<Input>();
    input->subscriber = std::make_unique<unitree::robot::ChannelSubscriber<LowCmd>>(topic);
    std::weak_ptr<Input> weak_input = input;
    input->subscriber->InitChannel([this, weak_input](const void* message) {
      const auto input = weak_input.lock();
      if (!input) return;
      const auto& command = *static_cast<const LowCmd*>(message);
      bool valid = true;
      for (const auto& motor : command.motor_cmd()) {
        valid = valid && std::isfinite(motor.q()) && std::isfinite(motor.dq()) &&
            std::isfinite(motor.tau()) && std::isfinite(motor.kp()) &&
            std::isfinite(motor.kd()) && motor.kp() >= 0.f && motor.kp() <= 1000.f &&
            motor.kd() >= 0.f && motor.kd() <= 100.f;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      input->valid = valid;
      input->command = command;
      input->stamp = Clock::now();
    }, 1);
    std::lock_guard<std::mutex> lock(mutex_);
    inputs_.emplace(topic, std::move(input));
  }

  bool ready(const std::string& topic) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto input = inputs_.find(topic);
    return input != inputs_.end() && fresh(*input->second);
  }

  bool select(const std::string& topic) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto input = inputs_.find(topic);
    if (input == inputs_.end() || !fresh(*input->second) || !publisher_.Write(input->second->command)) {
      return false;
    }
    selected_ = topic;
    return true;
  }

  void release(const std::string& topic) override {
    std::shared_ptr<Input> removed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (selected_ == topic) selected_.clear();
      const auto input = inputs_.find(topic);
      if (input != inputs_.end()) {
        removed = std::move(input->second);
        inputs_.erase(input);
      }
    }
    if (removed) removed->subscriber->CloseChannel();
  }

 private:
  struct Input {
    std::unique_ptr<unitree::robot::ChannelSubscriber<LowCmd>> subscriber;
    LowCmd command;
    Clock::time_point stamp{};
    bool valid{false};
  };

  bool fresh(const Input& input) const {
    const auto now = Clock::now();
    return input.valid && now - input.stamp < std::chrono::milliseconds(100) &&
        state_stamp_ != Clock::time_point{} && now - state_stamp_ < std::chrono::milliseconds(100);
  }

  std::string output_topic_;
  unitree::robot::ChannelPublisher<LowCmd> publisher_;
  unitree::robot::ChannelSubscriber<LowState> state_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Input>> inputs_;
  std::string selected_;
  Clock::time_point state_stamp_{};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

}

std::shared_ptr<PolicyCommandRelay> make_policy_command_relay(
    const std::string& output_topic, const std::string& state_topic) {
  return std::make_shared<DdsPolicyCommandRelay>(output_topic, state_topic);
}

}
