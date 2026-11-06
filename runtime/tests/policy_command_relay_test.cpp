#include "action_go/runtime/policy_command_relay.h"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool eventually(const std::function<bool()>& condition) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (condition()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

}

int main() {
  try {
    unitree::robot::ChannelFactory::Instance()->Init(85, "lo");
    const std::string prefix = "action_go/test/relay/" + std::to_string(getpid());
    std::atomic<int> observed{0};
    std::atomic<int> received{0};
    std::atomic<bool> stop{false};
    std::atomic<bool> state_enabled{true};
    std::atomic<bool> source_valid{true};
    unitree::robot::ChannelSubscriber<LowCmd> output(prefix + "/output");
    output.InitChannel([&](const void* message) {
      const auto& command = *static_cast<const LowCmd*>(message);
      observed.store(static_cast<int>(command.motor_cmd()[0].q()));
      received.fetch_add(1);
    }, 1);
    auto relay = action_go::runtime::make_policy_command_relay(prefix + "/output", prefix + "/state");
    relay->add(prefix + "/source");
    relay->add(prefix + "/target");
    unitree::robot::ChannelPublisher<LowState> state_publisher(prefix + "/state");
    unitree::robot::ChannelPublisher<LowCmd> source_publisher(prefix + "/source");
    unitree::robot::ChannelPublisher<LowCmd> target_publisher(prefix + "/target");
    state_publisher.InitChannel();
    source_publisher.InitChannel();
    target_publisher.InitChannel();
    std::thread producer([&]() {
      LowState state;
      state.imu_state().quaternion()[0] = 1.f;
      LowCmd source;
      LowCmd target;
      target.motor_cmd()[0].q() = 2.f;
      while (!stop.load()) {
        source.motor_cmd()[0].q() = source_valid.load() ? 1.f : std::numeric_limits<float>::quiet_NaN();
        if (state_enabled.load()) state_publisher.Write(state);
        source_publisher.Write(source);
        target_publisher.Write(target);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
    try {
      require(eventually([&]() { return relay->ready(prefix + "/source") && relay->ready(prefix + "/target"); }),
              "private routes did not become ready");
      require(received.load() == 0, "unselected commands reached output");
      require(relay->select(prefix + "/source"), "source selection failed");
      require(eventually([&]() { return observed.load() == 1; }), "selected source was not forwarded");
      require(relay->select(prefix + "/target"), "target selection failed");
      require(eventually([&]() { return observed.load() == 2; }), "selected target was not forwarded");
      relay->release(prefix + "/source");
      require(!relay->ready(prefix + "/source"), "released source remained ready");
      state_enabled.store(false);
      require(eventually([&]() { return !relay->ready(prefix + "/target"); }), "stale state remained ready");
      require(!relay->select(prefix + "/target"), "stale state allowed handoff");
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      const int count = received.load();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      require(received.load() == count, "relay continued output with stale state");
      state_enabled.store(true);
      require(eventually([&]() { return relay->ready(prefix + "/target"); }), "fresh state did not restore readiness");
      relay->add(prefix + "/source");
      require(eventually([&]() { return relay->ready(prefix + "/source"); }), "source did not become ready again");
      source_valid.store(false);
      require(eventually([&]() { return !relay->ready(prefix + "/source"); }), "invalid source remained ready");
      require(!relay->select(prefix + "/source"), "NaN command allowed handoff");
      relay->release(prefix + "/target");
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      const int released_count = received.load();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      require(received.load() == released_count, "released target continued output");
    } catch (...) {
      stop.store(true);
      producer.join();
      throw;
    }
    stop.store(true);
    producer.join();
    relay.reset();
    output.CloseChannel();
    std::cout << "Policy command relay tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
