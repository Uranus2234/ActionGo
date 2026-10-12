#include "action_go/motion_action_controller.h"

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace fs = std::filesystem;
namespace {
std::atomic<int> g_stop_level{0};

void emergency_signal_handler(int) {
  int expected = g_stop_level.load();
  if (expected >= 1) {
    g_stop_level.store(2);
    return;
  }
  while (expected < 1 && !g_stop_level.compare_exchange_weak(expected, 1)) {}
}

void damping_signal_handler(int) {
  g_stop_level.store(2);
}

struct Args {
  std::string net{"enP8p1s0"};
  std::string action{"kick_right"};
  std::string config;
  std::string event_log;
  double stop_after_sec{0.0};
  bool remote_control{false};
  bool dry_run{false};
};

enum RemoteButton : int {
  kR1 = 0, kL1 = 1, kStart = 2, kSelect = 3,
  kR2 = 4, kL2 = 5, kF1 = 6, kF2 = 7,
  kA = 8, kB = 9, kX = 10, kY = 11,
  kUp = 12, kRight = 13, kDown = 14, kLeft = 15
};

class RemoteMonitor {
 public:
  explicit RemoteMonitor(const std::string& topic = "rt/lowstate") {
    subscriber_.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(topic));
    subscriber_->InitChannel(std::bind(&RemoteMonitor::onLowState, this, std::placeholders::_1), 1);
  }

  bool connected() const { return connected_.load(); }

  bool comboPressed(std::initializer_list<int> buttons) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const int button : buttons) {
      if (button < 0 || button >= static_cast<int>(buttons_.size()) || !buttons_[button]) {
        return false;
      }
    }
    return true;
  }

  bool consumeComboRising(std::initializer_list<int> buttons) {
    const bool now = comboPressed(buttons);
    bool expected = false;
    if (now && combo_latch_.compare_exchange_strong(expected, true)) {
      return true;
    }
    if (!now) {
      combo_latch_.store(false);
    }
    return false;
  }

 private:
  void onLowState(const void* message) {
    const auto* low = static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
    const auto& remote = low->wireless_remote();
    if (remote.size() < 4) {
      return;
    }

    // Same mapping as RoboJuDo:
    // bit0..15 = R1,L1,Start,Select,R2,L2,F1,F2,A,B,X,Y,Up,Right,Down,Left
    const uint16_t keys = static_cast<uint16_t>(remote[2]) |
                          (static_cast<uint16_t>(remote[3]) << 8);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (int i = 0; i < 16; ++i) {
        buttons_[i] = ((keys >> i) & 0x1) != 0;
      }
    }
    connected_.store(true);
  }

  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> subscriber_;
  mutable std::mutex mutex_;
  std::array<bool, 16> buttons_{};
  std::atomic<bool> connected_{false};
  std::atomic<bool> combo_latch_{false};
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (k == "--net") args.net = need("--net");
    else if (k == "--action") args.action = need("--action");
    else if (k == "--config") args.config = need("--config");
    else if (k == "--event-log") args.event_log = need("--event-log");
    else if (k == "--stop-after-sec") args.stop_after_sec = std::stod(need("--stop-after-sec"));
    else if (k == "--remote-control") args.remote_control = true;
    else if (k == "--dry-run") args.dry_run = true;
    else if (k == "--help") {
      std::cout << "Usage: action_go_test --net enP8p1s0 --action kick_right [--config action_go/config/motion_actions.yaml] [--event-log events.jsonl] [--stop-after-sec SEC] [--dry-run]\n"
                << "Emergency keys while action is running:\n"
                << "  Ctrl+C       -> dampingStop(): force stop and enter official Damp\n"
                << "  Ctrl+\\       -> dampingStop(): force stop and enter official Damp\n"
                << "Remote mode:\n"
                << "  --remote-control waits for Start to start\n"
                << "  Select+Start -> dampingStop()\n"
                << "Dry-run:\n"
                << "  --dry-run does not StopMove/UserCtrl/switch official; players use no-LowCmd dry-run flags\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }

  if (args.config.empty()) {
    const fs::path cwd = fs::current_path();
    const fs::path c1 = cwd / "../action_go/config/motion_actions.yaml";
    const fs::path c2 = cwd / "../../action_go/config/motion_actions.yaml";
    for (const auto& p : {c1, c2}) {
      if (fs::exists(p)) {
        args.config = p.lexically_normal().string();
        break;
      }
    }
    if (args.config.empty()) args.config = c1.lexically_normal().string();
  }
  if (args.event_log.empty()) {
    const char* environment_log = std::getenv("ACTION_GO_EVENT_LOG");
    if (environment_log) args.event_log = environment_log;
  }
  if (args.stop_after_sec < 0.0) {
    throw std::runtime_error("--stop-after-sec must be non-negative");
  }
  return args;
}
} // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, emergency_signal_handler);
  std::signal(SIGTERM, emergency_signal_handler);
  std::signal(SIGQUIT, damping_signal_handler);

  try {
    const Args args = parse_args(argc, argv);
    std::cout << "[action_go_test] net=" << args.net
              << " action=" << args.action
              << " config=" << args.config;
    if (args.dry_run) std::cout << " DRY_RUN=1";
    std::cout << std::endl;

    unitree::robot::ChannelFactory::Instance()->Init(0, args.net);

    unitree::robot::g1::LocoClient loco;
    loco.Init();
    loco.SetTimeout(10.0f);

    MotionActionController controller;
    controller.setEventLogPath(args.event_log);
    if (!controller.init(&loco, args.config, args.net, args.dry_run)) {
      std::cerr << "[action_go_test] init failed: " << controller.lastError() << std::endl;
      return 1;
    }
    if (args.dry_run) {
      std::cout << "[action_go_test][dry-run] no StopMove/UserCtrl/official switch; "
                << "player command runs with no-LowCmd dry-run flag" << std::endl;
    }

    std::unique_ptr<RemoteMonitor> remote;
    if (args.remote_control) {
      remote = std::make_unique<RemoteMonitor>();
      std::cout << "[action_go_test] remote control enabled\n"
                << "  Start: start action\n"
                << "  Select+Start: dampingStop\n";

      while (!remote->connected()) {
        const int stop_level = g_stop_level.load();
        if (stop_level >= 2) {
          controller.dampingStop();
          return 3;
        }
        if (stop_level == 1) {
          controller.dampingStop();
          return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }

      std::cout << "[action_go_test] waiting for Start to start action '" << args.action << "'..." << std::endl;
      while (true) {
        const int stop_level = g_stop_level.load();
        if (stop_level >= 2) {
          controller.dampingStop();
          return 3;
        }
        if (stop_level == 1) {
          controller.dampingStop();
          return 2;
        }
        if (remote->consumeComboRising({kSelect, kStart})) {
          controller.dampingStop();
          return 3;
        }
        if (remote->consumeComboRising({kStart})) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      std::cout << "[action_go_test] remote start received" << std::endl;
    }

    if (!controller.startAction(args.action)) {
      std::cerr << "[action_go_test] startAction failed: " << controller.lastError() << std::endl;
      return 1;
    }

    std::optional<std::chrono::steady_clock::time_point> running_since;

    while (true) {
      if (remote) {
        if (remote->consumeComboRising({kSelect, kStart})) {
          g_stop_level.store(2);
        }
      }

      const int stop_level = g_stop_level.load();
      if (stop_level >= 1) {
        std::cerr << "[action_go_test] damping stop signal received" << std::endl;
        controller.dampingStop();
        return 3;
      }

      const auto status = controller.update();
      if (status == MotionActionController::Status::Running && !running_since) {
        running_since = std::chrono::steady_clock::now();
      }
      if (args.stop_after_sec > 0.0 && running_since &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - *running_since).count() >=
              args.stop_after_sec) {
        std::cout << "[action_go_test] stop-after reached; switch official before stopping payload"
                  << std::endl;
        if (!controller.stopAndSwitchToOfficial("walkrun")) {
          std::cerr << "[action_go_test] timed switch-back failed: "
                    << controller.lastError() << std::endl;
          return 1;
        }
        return 0;
      }
      if (status == MotionActionController::Status::Done) {
        std::cout << "[action_go_test] action done" << std::endl;
        return 0;
      }
      if (status == MotionActionController::Status::Failed) {
        std::cerr << "[action_go_test] action failed: " << controller.lastError() << std::endl;
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  } catch (const std::exception& e) {
    std::cerr << "[action_go_test] ERROR: " << e.what() << std::endl;
    return 1;
  }
}
