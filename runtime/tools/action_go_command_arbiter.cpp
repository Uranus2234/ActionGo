#include "action_go/runtime/candidate_command_arbiter.h"
#include "action_go/runtime/event_logger.h"
#include "action_go/runtime/yaml_contract.h"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <yaml-cpp/yaml.h>

#ifdef ACTION_GO_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using action_go::runtime::ArbiterRobotState;
using action_go::runtime::CandidateCommandArbiter;
using action_go::runtime::CandidateCommandArbiterConfig;
using action_go::runtime::CommandArbiterState;
using action_go::runtime::CommandSelection;
using action_go::runtime::JointCommandCandidate;
using action_go::runtime::RuntimeEventLogger;
using action_go::runtime::command_arbiter_state_name;
using action_go::runtime::command_selection_name;
using action_go::runtime::event_bool;
using action_go::runtime::event_int;
using action_go::runtime::event_number;
using action_go::runtime::event_string;
using action_go::runtime::SourceExitStatusV1;
using action_go::runtime::handoff_decision_name;

namespace {

std::atomic<bool> g_stop{false};

void signal_handler(int) {
  g_stop.store(true);
}

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xFFFFFFFF;
  constexpr uint32_t polynomial = 0x04c11db7;
  for (uint32_t index = 0; index < len; ++index) {
    uint32_t xbit = 1u << 31;
    const uint32_t data = ptr[index];
    for (uint32_t bit = 0; bit < 32; ++bit) {
      if (crc & 0x80000000) crc = (crc << 1) ^ polynomial;
      else crc <<= 1;
      if (data & xbit) crc ^= polynomial;
      xbit >>= 1;
    }
  }
  return crc;
}

struct Args {
  std::string net{"enP8p1s0"};
  int domain_id{0};
  std::string source_topic{"action_go/candidate/source"};
  std::string target_topic{"action_go/candidate/target"};
  std::string output_topic{"rt/user_lowcmd"};
  std::string state_topic{"rt/lowstate"};
  std::string event_log;
  std::string limits_yaml;
  std::string lower_key{"robot.joint_pos_lower_limit"};
  std::string upper_key{"robot.joint_pos_upper_limit"};
  std::string preview_file;
  std::string armed_file;
  std::string commit_file;
  std::string discard_file;
  std::string ready_file;
  std::string fallback_file;
  std::string source_exit_file;
  std::string hrm_onnx;
  std::string source_policy{"source"};
  std::string target_policy{"target"};
  int motor_count{29};
  int publish_hz{500};
  int stats_interval{250};
  double duration_sec{0.0};
  double candidate_timeout_sec{0.10};
  double lowstate_timeout_sec{0.20};
  double hrm_threshold{0.5};
  float reject_target_jump_rad{1.2f};
  float reject_target_velocity_jump{20.f};
  float reject_target_torque_jump{120.f};
  float dq_abs_max{30.f};
  float tau_abs_max{200.f};
  float hold_kp{20.f};
  float hold_kd{1.f};
  bool require_source_exit_gate{false};
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
    else if (key == "--domain-id") args.domain_id = std::stoi(need());
    else if (key == "--source-topic") args.source_topic = need();
    else if (key == "--target-topic") args.target_topic = need();
    else if (key == "--output-topic" || key == "--cmd-topic") args.output_topic = need();
    else if (key == "--state-topic") args.state_topic = need();
    else if (key == "--event-log") args.event_log = need();
    else if (key == "--limits-yaml") args.limits_yaml = need();
    else if (key == "--lower-key") args.lower_key = need();
    else if (key == "--upper-key") args.upper_key = need();
    else if (key == "--preview-file") args.preview_file = need();
    else if (key == "--armed-file") args.armed_file = need();
    else if (key == "--commit-file") args.commit_file = need();
    else if (key == "--discard-file") args.discard_file = need();
    else if (key == "--ready-file") args.ready_file = need();
    else if (key == "--fallback-file") args.fallback_file = need();
    else if (key == "--source-exit-file") args.source_exit_file = need();
    else if (key == "--hrm-onnx") args.hrm_onnx = need();
    else if (key == "--hrm-threshold") args.hrm_threshold = std::stod(need());
    else if (key == "--require-source-exit-gate") args.require_source_exit_gate = true;
    else if (key == "--source-policy") args.source_policy = need();
    else if (key == "--target-policy") args.target_policy = need();
    else if (key == "--motor-count") args.motor_count = std::stoi(need());
    else if (key == "--publish-hz") args.publish_hz = std::stoi(need());
    else if (key == "--stats-interval") args.stats_interval = std::stoi(need());
    else if (key == "--duration-sec") args.duration_sec = std::stod(need());
    else if (key == "--candidate-timeout-sec") args.candidate_timeout_sec = std::stod(need());
    else if (key == "--lowstate-timeout-sec") args.lowstate_timeout_sec = std::stod(need());
    else if (key == "--reject-target-position-jump-rad") args.reject_target_jump_rad = std::stof(need());
    else if (key == "--reject-target-velocity-jump") args.reject_target_velocity_jump = std::stof(need());
    else if (key == "--reject-target-torque-jump") args.reject_target_torque_jump = std::stof(need());
    else if (key == "--dq-abs-max") args.dq_abs_max = std::stof(need());
    else if (key == "--tau-abs-max") args.tau_abs_max = std::stof(need());
    else if (key == "--hold-kp") args.hold_kp = std::stof(need());
    else if (key == "--hold-kd") args.hold_kd = std::stof(need());
    else if (key == "--help" || key == "-h") {
      std::cout
          << "action_go_command_arbiter --source-topic TOPIC --target-topic TOPIC\n"
          << "  [--output-topic rt/user_lowcmd] [--state-topic rt/lowstate]\n"
          << "  [--preview-file PATH --armed-file PATH --commit-file PATH --discard-file PATH]\n"
          << "  [--source-exit-file PATH --require-source-exit-gate]\n"
          << "  [--hrm-onnx MODEL.onnx --hrm-threshold 0.5]\n"
          << "  [--limits-yaml PATH --lower-key KEY --upper-key KEY]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.motor_count <= 0 || args.publish_hz <= 0 || args.stats_interval <= 0) {
    throw std::runtime_error("motor-count, publish-hz, and stats-interval must be positive");
  }
  return args;
}

#ifdef ACTION_GO_HAS_ONNX
class HrmOnnxEvaluator {
 public:
  explicit HrmOnnxEvaluator(const std::string& path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "action_go_hrm"),
        session_(nullptr),
        memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    session_ = Ort::Session(env_, path.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
  }

  std::optional<double> infer(
      const action_go::runtime::HrmFeatureVectorV1& features) {
    if (features.schema_version != action_go::runtime::kHrmFeatureSchemaVersion ||
        features.values.empty()) {
      return std::nullopt;
    }
    std::vector<float> input = features.values;
    const std::array<int64_t, 2> shape{
        1, static_cast<int64_t>(input.size())};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_, input.data(), input.size(), shape.data(), shape.size());
    const char* input_name = input_name_.c_str();
    const char* output_name = output_name_.c_str();
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1,
                                &output_name, 1);
    if (outputs.empty() || !outputs[0].IsTensor() ||
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != 1) {
      return std::nullopt;
    }
    const float probability = outputs[0].GetTensorData<float>()[0];
    if (!std::isfinite(probability) || probability < 0.f || probability > 1.f) {
      return std::nullopt;
    }
    return static_cast<double>(probability);
  }

 private:
  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo memory_;
  std::string input_name_;
  std::string output_name_;
};
#endif

void touch_file(const std::string& path, const std::string& value) {
  if (path.empty()) return;
  fs::path output(path);
  if (output.has_parent_path()) fs::create_directories(output.parent_path());
  const fs::path temporary = output.string() + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::out | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to write lifecycle file: " + path);
    stream << value << '\n';
  }
  fs::rename(temporary, output);
}

void remove_file(const std::string& path) {
  if (path.empty()) return;
  std::error_code error;
  fs::remove(path, error);
}

std::string trim_copy(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

bool parse_bool_value(const std::string& value, bool& output) {
  const std::string normalized = trim_copy(value);
  if (normalized == "1" || normalized == "true" || normalized == "yes") {
    output = true;
    return true;
  }
  if (normalized == "0" || normalized == "false" || normalized == "no") {
    output = false;
    return true;
  }
  return false;
}

SourceExitStatusV1 read_source_exit_status(const std::string& path) {
  SourceExitStatusV1 status;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("failed to open source exit status: " + path);
  std::string line;
  while (std::getline(input, line)) {
    line = trim_copy(line);
    if (line.empty() || line[0] == '#') continue;
    const auto separator = line.find('=');
    if (separator == std::string::npos) continue;
    const std::string key = trim_copy(line.substr(0, separator));
    const std::string value = trim_copy(line.substr(separator + 1));
    if (key == "schema_version") status.schema_version = std::stoi(value);
    else if (key == "sequence") status.sequence = static_cast<uint64_t>(std::stoull(value));
    else if (key == "source_timestamp_sec") status.source_timestamp_sec = std::stod(value);
    else if (key == "received_timestamp_sec") status.received_timestamp_sec = std::stod(value);
    else if (key == "freshness_sec") status.freshness_sec = std::stod(value);
    else if (key == "valid") {
      if (!parse_bool_value(value, status.valid)) throw std::runtime_error("invalid source exit valid");
    } else if (key == "phase") status.phase = value;
    else if (key == "contact_mode") status.contact_mode = value;
    else if (key == "contact_count") status.contact_count = std::stoi(value);
    else if (key == "airborne") {
      if (!parse_bool_value(value, status.airborne)) throw std::runtime_error("invalid source exit airborne");
    } else if (key == "rotation") {
      if (!parse_bool_value(value, status.rotation)) throw std::runtime_error("invalid source exit rotation");
    } else if (key == "fallen") {
      if (!parse_bool_value(value, status.fallen)) throw std::runtime_error("invalid source exit fallen");
    } else if (key == "base_height_m") status.base_height_m = std::stod(value);
    else if (key == "tilt_deg") status.tilt_deg = std::stod(value);
    else if (key == "angular_speed_rps") status.angular_speed_rps = std::stod(value);
    else if (key == "retry_policy") status.retry_policy = value;
  }
  return status;
}

CandidateCommandArbiterConfig arbiter_config(const Args& args) {
  CandidateCommandArbiterConfig config;
  config.motor_count = static_cast<size_t>(args.motor_count);
  config.candidate_timeout_sec = args.candidate_timeout_sec;
  config.lowstate_timeout_sec = args.lowstate_timeout_sec;
  config.reject_target_position_jump_rad = args.reject_target_jump_rad;
  config.reject_target_velocity_jump = args.reject_target_velocity_jump;
  config.reject_target_torque_jump = args.reject_target_torque_jump;
  config.dq_abs_max = args.dq_abs_max;
  config.tau_abs_max = args.tau_abs_max;
  config.hold_kp = args.hold_kp;
  config.hold_kd = args.hold_kd;
  config.source_exit_gate.required = args.require_source_exit_gate || !args.source_exit_file.empty();
  if (!args.limits_yaml.empty()) {
    const YAML::Node yaml = YAML::LoadFile(args.limits_yaml);
    config.lower_limits = action_go::runtime::yaml_contract::required_vector_size<float>(
        yaml, args.lower_key, config.motor_count);
    config.upper_limits = action_go::runtime::yaml_contract::required_vector_size<float>(
        yaml, args.upper_key, config.motor_count);
  }
  return config;
}

JointCommandCandidate from_lowcmd(const LowCmd& message, size_t motor_count,
                                  uint64_t sequence) {
  JointCommandCandidate command;
  command.q.resize(motor_count);
  command.dq.resize(motor_count);
  command.tau.resize(motor_count);
  command.kp.resize(motor_count);
  command.kd.resize(motor_count);
  if (message.motor_cmd().size() < motor_count) {
    command.q.clear();
    command.stamp = std::chrono::steady_clock::now();
    command.sequence = sequence;
    return command;
  }
  for (size_t index = 0; index < motor_count; ++index) {
    const auto& motor = message.motor_cmd()[index];
    command.q[index] = motor.q();
    command.dq[index] = motor.dq();
    command.tau[index] = motor.tau();
    command.kp[index] = motor.kp();
    command.kd[index] = motor.kd();
  }
  command.mode_machine = static_cast<int>(message.mode_machine());
  command.sequence = sequence;
  command.stamp = std::chrono::steady_clock::now();
  return command;
}

ArbiterRobotState from_lowstate(const LowState& message, size_t motor_count,
                                uint64_t sequence) {
  ArbiterRobotState state;
  if (message.motor_state().size() >= motor_count) {
    state.q.resize(motor_count);
    state.dq.resize(motor_count);
    for (size_t index = 0; index < motor_count; ++index) {
      state.q[index] = message.motor_state()[index].q();
      state.dq[index] = message.motor_state()[index].dq();
    }
    state.connected = true;
  }
  state.sequence = sequence;
  state.mode_machine = static_cast<int>(message.mode_machine());
  state.stamp = std::chrono::steady_clock::now();
  return state;
}

LowCmd to_lowcmd(const JointCommandCandidate& command, size_t motor_count) {
  LowCmd message;
  message.mode_pr() = 0;
  message.mode_machine() = static_cast<uint8_t>(command.mode_machine);
  for (size_t index = 0; index < motor_count && index < message.motor_cmd().size(); ++index) {
    auto& motor = message.motor_cmd()[index];
    motor.mode() = 1;
    motor.q() = command.q[index];
    motor.dq() = command.dq[index];
    motor.tau() = command.tau[index];
    motor.kp() = command.kp[index];
    motor.kd() = command.kd[index];
  }
  message.crc() = crc32_core(reinterpret_cast<uint32_t*>(&message),
                             (sizeof(message) >> 2) - 1);
  return message;
}

class DdsArbiter {
 public:
  explicit DdsArbiter(const Args& args)
      : args_(args), arbiter_(arbiter_config(args)) {
    logger_.open(args.event_log);
    const auto context = logger_.context();
    arbiter_.set_context(context.run_id, context.trial_id, context.pair_id,
                         args.source_policy, args.target_policy);
    if (!args.hrm_onnx.empty()) {
#ifdef ACTION_GO_HAS_ONNX
      hrm_evaluator_ = std::make_shared<HrmOnnxEvaluator>(args.hrm_onnx);
      arbiter_.set_hrm_risk_evaluator(
          [model = hrm_evaluator_](
              const action_go::runtime::HrmFeatureVectorV1& features) {
            return model->infer(features);
          },
          args.hrm_threshold);
      logger_.log("hrm_model_loaded",
                  {event_string("model", args.hrm_onnx),
                   event_number("authorization_threshold", args.hrm_threshold)});
#else
      throw std::runtime_error(
          "--hrm-onnx requested but action_go_command_arbiter was built without ONNX Runtime");
#endif
    }
  }

  void on_source(const LowCmd& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    arbiter_.update_source(from_lowcmd(message, static_cast<size_t>(args_.motor_count),
                                       ++source_sequence_));
  }

  void on_target(const LowCmd& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    arbiter_.update_target(from_lowcmd(message, static_cast<size_t>(args_.motor_count),
                                       ++target_sequence_));
  }

  void on_lowstate(const LowState& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    arbiter_.update_lowstate(from_lowstate(message, static_cast<size_t>(args_.motor_count),
                                           ++lowstate_sequence_));
  }

  void poll_files() {
    poll_source_exit_status();
    if (!preview_seen_ && !args_.preview_file.empty() && fs::exists(args_.preview_file)) {
      std::lock_guard<std::mutex> lock(mutex_);
      arbiter_.request_preview();
      preview_seen_ = true;
      logger_.log("preview_requested",
                  {event_string("source_policy", args_.source_policy),
                   event_string("target_policy", args_.target_policy)});
    }
    if (!commit_seen_ && !args_.commit_file.empty() && fs::exists(args_.commit_file)) {
      std::lock_guard<std::mutex> lock(mutex_);
      arbiter_.request_commit();
      commit_seen_ = true;
      logger_.log("commit_requested");
    }
    if (!discard_seen_ && !args_.discard_file.empty() && fs::exists(args_.discard_file)) {
      std::lock_guard<std::mutex> lock(mutex_);
      arbiter_.request_discard();
      discard_seen_ = true;
      logger_.log("discard_requested");
    }
  }

  action_go::runtime::ArbiterOutput tick() {
    std::lock_guard<std::mutex> lock(mutex_);
    return arbiter_.tick();
  }

  RuntimeEventLogger& logger() { return logger_; }

 private:
  void poll_source_exit_status() {
    if (args_.source_exit_file.empty() || !fs::exists(args_.source_exit_file)) return;
    std::error_code error;
    const auto modified = fs::last_write_time(args_.source_exit_file, error);
    if (error) return;
    try {
      if (!source_exit_mtime_ || modified != *source_exit_mtime_) {
        source_exit_status_ = read_source_exit_status(args_.source_exit_file);
        source_exit_mtime_ = modified;
      }
      if (!source_exit_status_) return;
      auto status = *source_exit_status_;
      status.freshness_sec = std::max(
          0.0, std::chrono::duration<double>(fs::file_time_type::clock::now() - modified).count());
      std::lock_guard<std::mutex> lock(mutex_);
      arbiter_.update_source_exit_status(status);
      if (!source_exit_logged_sequence_ ||
          *source_exit_logged_sequence_ != status.sequence) {
        source_exit_logged_sequence_ = status.sequence;
        logger_.log("source_exit_status_updated",
                    {event_int("sequence", static_cast<long long>(status.sequence)),
                     event_number("freshness_sec", status.freshness_sec),
                     event_string("phase", status.phase),
                     event_string("contact_mode", status.contact_mode),
                     event_int("contact_count", status.contact_count),
                     event_bool("airborne", status.airborne),
                     event_bool("rotation", status.rotation),
                     event_bool("fallen", status.fallen),
                     event_number("base_height_m", status.base_height_m),
                     event_number("tilt_deg", status.tilt_deg),
                     event_number("angular_speed_rps", status.angular_speed_rps)});
      }
    } catch (const std::exception& error_value) {
      logger_.log("source_exit_status_invalid",
                  {event_string("reason", error_value.what())});
    }
  }

  Args args_;
  CandidateCommandArbiter arbiter_;
  RuntimeEventLogger logger_;
  std::mutex mutex_;
  uint64_t source_sequence_{0};
  uint64_t target_sequence_{0};
  uint64_t lowstate_sequence_{0};
  bool preview_seen_{false};
  bool commit_seen_{false};
  bool discard_seen_{false};
  std::optional<fs::file_time_type> source_exit_mtime_;
  std::optional<SourceExitStatusV1> source_exit_status_;
  std::optional<uint64_t> source_exit_logged_sequence_;
#ifdef ACTION_GO_HAS_ONNX
  std::shared_ptr<HrmOnnxEvaluator> hrm_evaluator_;
#endif
};

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    const Args args = parse_args(argc, argv);
    for (const std::string* path : {&args.preview_file, &args.armed_file, &args.commit_file,
                                    &args.discard_file, &args.ready_file, &args.fallback_file}) {
      remove_file(*path);
    }

    unitree::robot::ChannelFactory::Instance()->Init(args.domain_id, args.net);
    DdsArbiter runtime(args);
    unitree::robot::ChannelPublisher<LowCmd> publisher(args.output_topic);
    publisher.InitChannel();
    unitree::robot::ChannelSubscriber<LowCmd> source(args.source_topic);
    unitree::robot::ChannelSubscriber<LowCmd> target(args.target_topic);
    unitree::robot::ChannelSubscriber<LowState> lowstate(args.state_topic);
    source.InitChannel([&runtime](const void* message) {
      runtime.on_source(*static_cast<const LowCmd*>(message));
    }, 1);
    target.InitChannel([&runtime](const void* message) {
      runtime.on_target(*static_cast<const LowCmd*>(message));
    }, 1);
    lowstate.InitChannel([&runtime](const void* message) {
      runtime.on_lowstate(*static_cast<const LowState*>(message));
    }, 1);

    touch_file(args.ready_file, "ready");
    runtime.logger().log(
        "command_arbiter_started",
        {event_string("source_topic", args.source_topic),
         event_string("target_topic", args.target_topic),
         event_string("output_topic", args.output_topic),
         event_int("publish_hz", args.publish_hz)});

    const auto start_time = std::chrono::steady_clock::now();
    const auto publish_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / args.publish_hz));
    auto next_publish = start_time;
    CommandArbiterState last_state = CommandArbiterState::kSourceActive;
    CommandSelection last_selection = CommandSelection::kNone;
    uint64_t tick_count = 0;
    bool armed_written = false;
    bool fallback_written = false;

    while (!g_stop.load()) {
      if (args.duration_sec > 0.0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() >=
              args.duration_sec) {
        break;
      }
      runtime.poll_files();
      const auto output = runtime.tick();
      ++tick_count;
      if (output.publish) publisher.Write(to_lowcmd(output.command, args.motor_count));

      if (output.state != last_state || output.selection != last_selection || output.fallback) {
        runtime.logger().log(
            "command_arbiter_transition",
            {event_string("state", command_arbiter_state_name(output.state)),
             event_string("selection", command_selection_name(output.selection)),
             event_bool("publish", output.publish),
             event_bool("fallback", output.fallback),
             event_string("reason", output.reason),
             event_number("command_jump_rad", output.metrics.max_position_jump_rad),
             event_number("velocity_jump", output.metrics.max_velocity_jump),
             event_number("torque_jump", output.metrics.max_torque_jump),
             event_number("command_rate", output.metrics.max_position_rate),
             event_number("command_jerk", output.metrics.max_position_jerk),
             event_number("source_age_sec", output.metrics.source_age_sec),
             event_number("target_age_sec", output.metrics.target_age_sec)});
        runtime.logger().log(
            "handoff_assessment",
            {event_int("assessment_schema_version", output.assessment.schema_version),
             event_string("source_policy", output.assessment.source_policy),
             event_string("target_policy", output.assessment.target_policy),
             event_string("group", output.assessment.group),
             event_int("source_sequence",
                       static_cast<long long>(output.assessment.source_sequence)),
             event_int("target_sequence",
                       static_cast<long long>(output.assessment.target_sequence)),
             event_bool("hard_preconditions_met",
                        output.assessment.hard_preconditions_met),
             event_bool("authorized", output.assessment.authorized),
             event_string("decision", handoff_decision_name(output.assessment.decision)),
             event_string("source_phase", output.assessment.source_phase),
             event_bool("source_exit_admissible", output.assessment.source_exit_admissible),
             event_bool("target_viability", output.assessment.target_viability),
             event_string("retry_policy", output.assessment.retry_policy),
             event_string("reject_reason", output.assessment.reject_reason),
             event_number("failure_probability",
                          output.assessment.failure_probability),
             event_number("uncertainty", output.assessment.uncertainty),
             event_number("target_age_sec", output.assessment.target_age_sec),
             event_number("max_position_jump_rad",
                          output.assessment.max_position_jump_rad),
             event_number("max_velocity_jump",
                          output.assessment.max_velocity_jump),
             event_number("max_torque_jump",
                          output.assessment.max_torque_jump),
             event_string("reason", output.assessment.reason)});
      }
      if (!armed_written && output.state == CommandArbiterState::kTargetArmed) {
        touch_file(args.armed_file, "armed");
        armed_written = true;
        runtime.logger().log("target_armed",
                             {event_int("source_sequence", output.assessment.source_sequence),
                              event_int("target_sequence", output.assessment.target_sequence),
                              event_number("command_jump_rad",
                                           output.assessment.max_position_jump_rad)});
      }
      if (!fallback_written && output.fallback) {
        touch_file(args.fallback_file, output.reason);
        fallback_written = true;
        runtime.logger().log("fault_detected",
                             {event_string("reason", output.reason),
                              event_string("state", command_arbiter_state_name(output.state)),
                              event_string("selection", command_selection_name(output.selection))});
        runtime.logger().log("fallback_completed",
                             {event_string("reason", output.reason),
                              event_string("selection", command_selection_name(output.selection))});
      }
      if (tick_count % static_cast<uint64_t>(args.stats_interval) == 0) {
        runtime.logger().log(
            "command_arbiter_metrics",
            {event_int("tick", static_cast<long long>(tick_count)),
             event_string("state", command_arbiter_state_name(output.state)),
             event_string("selection", command_selection_name(output.selection)),
             event_number("command_jump_rad", output.metrics.max_position_jump_rad),
             event_number("command_rate", output.metrics.max_position_rate),
             event_number("command_jerk", output.metrics.max_position_jerk),
             event_number("source_age_sec", output.metrics.source_age_sec),
             event_number("target_age_sec", output.metrics.target_age_sec)});
      }
      last_state = output.state;
      last_selection = output.selection;
      next_publish += publish_period;
      if (next_publish < std::chrono::steady_clock::now()) {
        next_publish = std::chrono::steady_clock::now();
      }
      std::this_thread::sleep_until(next_publish);
    }
    runtime.logger().log("command_arbiter_stopped",
                         {event_int("ticks", static_cast<long long>(tick_count))});
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[action_go_command_arbiter] " << error.what() << '\n';
    return 1;
  }
}
