#include "action_go/runtime/event_logger.h"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

namespace {

std::atomic<bool> g_stop{false};

void signal_handler(int) {
  g_stop.store(true);
}

std::string json_escape(const std::string& value) {
  std::string out;
  for (char character : value) {
    switch (character) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += character; break;
    }
  }
  return out;
}

struct Args {
  std::string net{"enP8p1s0"};
  std::string lowcmd_topic{"rt/user_lowcmd"};
  std::string lowstate_topic{"rt/lowstate"};
  std::string output;
  double duration_sec{0.0};
  int sample_every{1};
  int userctrl_mode_machine{-1};
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    auto need = [&]() -> std::string {
      if (index + 1 >= argc) throw std::runtime_error("missing value for " + key);
      return argv[++index];
    };
    if (key == "--net") args.net = need();
    else if (key == "--lowcmd-topic" || key == "--cmd-topic") args.lowcmd_topic = need();
    else if (key == "--lowstate-topic" || key == "--state-topic") args.lowstate_topic = need();
    else if (key == "--output" || key == "--event-log") args.output = need();
    else if (key == "--duration-sec") args.duration_sec = std::stod(need());
    else if (key == "--sample-every") args.sample_every = std::stoi(need());
    else if (key == "--userctrl-mode-machine") args.userctrl_mode_machine = std::stoi(need());
    else if (key == "--help" || key == "-h") {
      std::cout << "action_go_trace_recorder --output trace.jsonl [--net NIC] "
                   "[--duration-sec SEC] [--sample-every N] "
                   "[--userctrl-mode-machine ID]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.output.empty()) throw std::runtime_error("--output is required");
  if (args.sample_every <= 0) throw std::runtime_error("--sample-every must be positive");
  return args;
}

class TraceWriter {
 public:
  TraceWriter(const std::string& path, int sample_every, int userctrl_mode_machine)
      : sample_every_(sample_every),
        userctrl_mode_machine_(userctrl_mode_machine),
        context_(action_go::runtime::action_lifecycle_context_from_environment()) {
    fs::path output_path(path);
    if (output_path.has_parent_path()) fs::create_directories(output_path.parent_path());
    output_.open(path, std::ios::out | std::ios::trunc);
    if (!output_) throw std::runtime_error("failed to open trace output: " + path);
    write_event("trace_recorder_started", "", 0, 0, true, 0.0, 0.0, 0.0);
  }

  void on_lowcmd(const LowCmd& command) {
    const auto now = std::chrono::steady_clock::now();
    const uint64_t sequence = ++lowcmd_count_;
    const long long gap_ns = last_lowcmd_.time_since_epoch().count() == 0
        ? 0
        : std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_lowcmd_).count();
    last_lowcmd_ = now;
    if (sequence != 1 && sequence % static_cast<uint64_t>(sample_every_) != 0) return;
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    double kp_max = 0.0;
    bool finite = true;
    for (const auto& motor : command.motor_cmd()) {
      q_min = std::min(q_min, static_cast<double>(motor.q()));
      q_max = std::max(q_max, static_cast<double>(motor.q()));
      kp_max = std::max(kp_max, std::abs(static_cast<double>(motor.kp())));
      finite = finite && std::isfinite(motor.q()) && std::isfinite(motor.kp()) &&
               std::isfinite(motor.kd()) && std::isfinite(motor.tau());
    }
    write_event("trace_lowcmd", "lowcmd", sequence, gap_ns, finite, q_min, q_max, kp_max,
                static_cast<int>(command.mode_machine()));
  }

  void on_lowstate(const LowState& state) {
    const uint64_t sequence = ++lowstate_count_;
    const int mode_machine = static_cast<int>(state.mode_machine());
    const int no_previous_mode = std::numeric_limits<int>::min();
    const int previous_mode = last_mode_machine_.exchange(mode_machine);
    if (previous_mode == no_previous_mode || previous_mode != mode_machine) {
      write_event(previous_mode == no_previous_mode
                      ? "trace_control_mode_observed"
                      : "trace_control_mode_changed",
                  "lowstate", sequence, 0, true, 0.0, 0.0, 0.0,
                  mode_machine, previous_mode,
                  userctrl_mode_machine_ >= 0 && mode_machine == userctrl_mode_machine_);
    }
    if (sequence != 1 && sequence % static_cast<uint64_t>(sample_every_) != 0) return;
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    bool finite = true;
    for (const auto& motor : state.motor_state()) {
      q_min = std::min(q_min, static_cast<double>(motor.q()));
      q_max = std::max(q_max, static_cast<double>(motor.q()));
      finite = finite && std::isfinite(motor.q()) && std::isfinite(motor.dq());
    }
    write_event("trace_lowstate", "lowstate", sequence, 0, finite, q_min, q_max, 0.0,
                mode_machine);
  }

  void finish() {
    write_event("trace_recorder_summary", "", 0, 0, true,
                static_cast<double>(lowcmd_count_.load()),
                static_cast<double>(lowstate_count_.load()), 0.0);
    std::lock_guard<std::mutex> lock(mutex_);
    output_.flush();
  }

 private:
  void write_event(const char* event, const char* stream, uint64_t sequence,
                   long long gap_ns, bool finite, double value_a,
                   double value_b, double value_c, int mode_machine = 0,
                   int previous_mode_machine = 0, bool is_userctrl = false) {
    const auto steady_now = std::chrono::steady_clock::now();
    const auto unix_now = std::chrono::system_clock::now();
    const auto steady_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        steady_now.time_since_epoch()).count();
    const auto unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        unix_now.time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(mutex_);
    output_ << "{\"event\":\"" << event << "\""
            << ",\"schema_version\":" << context_.schema_version
            << ",\"t_steady_ns\":" << steady_ns
            << ",\"t_unix_ns\":" << unix_ns
            << ",\"run_id\":\"" << json_escape(context_.run_id) << "\""
            << ",\"trial_id\":\"" << json_escape(context_.trial_id) << "\""
            << ",\"pair_id\":\"" << json_escape(context_.pair_id) << "\""
            << ",\"policy\":\"" << json_escape(context_.policy) << "\""
            << ",\"handoff_method\":\"" << json_escape(context_.handoff_method) << "\""
            << ",\"git_sha\":\"" << json_escape(context_.git_sha) << "\""
            << ",\"spec_sha256\":\"" << json_escape(context_.spec_sha256) << "\""
            << ",\"model_sha256\":\"" << json_escape(context_.model_sha256) << "\""
            << ",\"hardware_id\":\"" << json_escape(context_.hardware_id) << "\""
            << ",\"manifest_sha256\":\"" << json_escape(context_.manifest_sha256) << "\""
            << ",\"stream\":\"" << stream << "\""
            << ",\"sample_index\":" << sequence
            << ",\"gap_ns\":" << gap_ns
            << ",\"finite\":" << (finite ? "true" : "false")
            << ",\"value_a\":" << std::setprecision(9) << value_a
            << ",\"value_b\":" << value_b
            << ",\"value_c\":" << value_c
            << ",\"mode_machine\":" << mode_machine
            << ",\"previous_mode_machine\":" << previous_mode_machine
            << ",\"is_userctrl\":" << (is_userctrl ? "true" : "false") << "}\n";
  }

  int sample_every_{1};
  int userctrl_mode_machine_{-1};
  action_go::runtime::ActionLifecycleEventV1 context_;
  std::ofstream output_;
  std::mutex mutex_;
  std::atomic<uint64_t> lowcmd_count_{0};
  std::atomic<uint64_t> lowstate_count_{0};
  std::atomic<int> last_mode_machine_{std::numeric_limits<int>::min()};
  std::chrono::steady_clock::time_point last_lowcmd_{};
};

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    const Args args = parse_args(argc, argv);
    unitree::robot::ChannelFactory::Instance()->Init(0, args.net);
    TraceWriter writer(args.output, args.sample_every, args.userctrl_mode_machine);
    unitree::robot::ChannelSubscriber<LowCmd> lowcmd(args.lowcmd_topic);
    unitree::robot::ChannelSubscriber<LowState> lowstate(args.lowstate_topic);
    lowcmd.InitChannel([&writer](const void* message) {
      writer.on_lowcmd(*static_cast<const LowCmd*>(message));
    }, 1);
    lowstate.InitChannel([&writer](const void* message) {
      writer.on_lowstate(*static_cast<const LowState*>(message));
    }, 1);
    const auto start = std::chrono::steady_clock::now();
    while (!g_stop.load()) {
      if (args.duration_sec > 0.0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >=
              args.duration_sec) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    writer.finish();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[action_go_trace_recorder] " << error.what() << '\n';
    return 1;
  }
}
