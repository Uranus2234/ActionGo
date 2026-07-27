#include "action_go/runtime/command_safety_filter.h"
#include "action_go/runtime/event_logger.h"
#include "action_go/runtime/policy_protocol.h"
#include "action_go/runtime/preview_lifecycle.h"
#include "action_go/runtime/yaml_contract.h"

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#ifdef ACTION_GO_WITH_REALSENSE
#include <librealsense2/rs.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using IMUState = unitree_hg::msg::dds_::IMUState_;

namespace {
std::atomic<bool> g_stop_requested{false};
void signal_handler(int) { g_stop_requested.store(true); }

using action_go::runtime::CommandSafetyConfig;
using action_go::runtime::CommandSafetyFilter;
using action_go::runtime::CommandSafetyLimits;
using action_go::runtime::CommandSafetyState;
using action_go::runtime::RuntimeEventLogger;
using action_go::runtime::event_int;
using action_go::runtime::event_number;
using action_go::runtime::event_string;
using action_go::runtime::kUnitreeG1ProtocolMotorCount;
namespace yamlc = action_go::runtime::yaml_contract;

constexpr int kNumJoints = kUnitreeG1ProtocolMotorCount;
constexpr int kDefaultModeMachine = 5;
constexpr int kRemoteR1 = 0;
constexpr int kRemoteL1 = 1;
constexpr float kPi = 3.14159265358979323846f;

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0, data = 0, CRC32 = 0xFFFFFFFF;
  constexpr uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    xbit = 1u << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; ++bits) {
      if (CRC32 & 0x80000000) { CRC32 <<= 1; CRC32 ^= dwPolynomial; }
      else { CRC32 <<= 1; }
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) oss << ",";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::vector<float> tensor_to_vec_exact(const Ort::Value& value,
                                       size_t expected,
                                       const std::string& name) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  const size_t actual = info.GetElementCount();
  if (actual != expected) {
    throw std::runtime_error(name + " output dim mismatch: expected " + std::to_string(expected) +
                             " floats, got " + std::to_string(actual) +
                             " shape=" + shape_string(shape));
  }
  const float* p = value.GetTensorData<float>();
  std::vector<float> out(p, p + expected);
  for (size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i])) {
      throw std::runtime_error(name + " output contains NaN/Inf at index " + std::to_string(i));
    }
  }
  return out;
}

struct Args {
  std::string logdir;
  std::string net{"enP8p1s0"};
  std::string cmd_topic{"rt/user_lowcmd"};
  std::string state_topic{"rt/lowstate"};
  std::string torso_imu_topic{"rt/secondary_imu"};
  std::string depth_source{"realsense"};
  std::string command_source{"remote"};
  std::string standdir;
  std::string ready_file;
  std::string start_file;
  std::string preview_file;
  std::string commit_file;
  std::string discard_file;
  std::string event_log;
  int rs_width{480};
  int rs_height{270};
  int rs_fps{60};
  int hz{50};
  int publish_hz{500};
  int onnx_threads{2};
  int mode_machine{kDefaultModeMachine};
  int stats_interval{100};
  double target_interp_sec{0.02};
  double preview_timeout_sec{5.0};
  double target_rate_limit{0.0};
  double startup_sec{2.0};
  double startup_step_size{0.0};
  double startup_hold_sec{0.5};
  double duration_sec{0.0};
  double cold_start_step_size{0.2};
  double cold_start_hold_sec{0.0};
  float cold_start_kpkd_factor{2.0f};
  float kp_factor{1.0f};
  float kd_factor{1.0f};
  float joint_pos_protect_ratio{2.0f};
  bool clip_by_torque{true};
  bool use_normalizer{false};
  bool no_damping_on_stop{false};
  bool check_only{false};
  bool dry_run{false};
  bool dds_dry_run{false};
  bool camera_check{false};
  bool depth_camera_check_only{false};
  bool require_depth_camera{false};
  bool use_torso_imu{true};
  bool require_torso_imu{false};
  bool cold_start_before_stand{false};
  bool stand_until_l1{false};
  bool fail_on_bad_input{false};
  bool experimental_no_prestart_hold{false};
  double depth_camera_timeout_sec{1.0};
  double torso_imu_timeout_sec{1.0};
  double stand_warmup_sec{0.0};
  int min_depth_camera_frames{1};
  float max_preflight_tilt_deg{20.f};
  float max_preflight_dq{2.f};
  float max_preflight_action_abs{5.f};
  float max_preflight_target_err{2.f};
  int min_preflight_rs_frames{10};
  float lin_vel_deadband{0.5f};
  std::array<float, 2> lin_vel_range{0.5f, 0.5f};
  std::array<float, 2> lin_vel_negative_range{0.f, 0.f};
  std::array<float, 2> lat_vel_range{0.f, 0.f};
  std::array<float, 2> lat_vel_negative_range{0.f, 0.f};
  float ang_vel_deadband{0.5f};
  std::array<float, 2> ang_vel_range{0.f, 1.f};
  std::array<float, 2> ang_vel_negative_range{0.f, 1.f};
};

std::string need_arg(int& i, int argc, char** argv, const std::string& key) {
  if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
  return argv[++i];
}
std::array<float, 2> parse_pair(int& i, int argc, char** argv, const std::string& key) {
  if (i + 2 >= argc) throw std::runtime_error("missing 2 values for " + key);
  return {std::stof(argv[++i]), std::stof(argv[++i])};
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--logdir") a.logdir = need_arg(i, argc, argv, k);
    else if (k == "--net") a.net = need_arg(i, argc, argv, k);
    else if (k == "--cmd-topic") a.cmd_topic = need_arg(i, argc, argv, k);
    else if (k == "--state-topic") a.state_topic = need_arg(i, argc, argv, k);
    else if (k == "--torso-imu-topic") a.torso_imu_topic = need_arg(i, argc, argv, k);
    else if (k == "--depth-source") a.depth_source = need_arg(i, argc, argv, k);
    else if (k == "--command-source") a.command_source = need_arg(i, argc, argv, k);
    else if (k == "--standdir" || k == "--stand-logdir") a.standdir = need_arg(i, argc, argv, k);
    else if (k == "--ready-file") a.ready_file = need_arg(i, argc, argv, k);
    else if (k == "--start-file") a.start_file = need_arg(i, argc, argv, k);
    else if (k == "--preview-file") a.preview_file = need_arg(i, argc, argv, k);
    else if (k == "--commit-file") a.commit_file = need_arg(i, argc, argv, k);
    else if (k == "--discard-file") a.discard_file = need_arg(i, argc, argv, k);
    else if (k == "--event-log") a.event_log = need_arg(i, argc, argv, k);
    else if (k == "--rs-resolution") { a.rs_width = std::stoi(need_arg(i, argc, argv, k)); a.rs_height = std::stoi(need_arg(i, argc, argv, k)); }
    else if (k == "--rs-fps") a.rs_fps = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--hz") a.hz = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--publish-hz") a.publish_hz = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--onnx-threads") a.onnx_threads = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--target-interp-sec") a.target_interp_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--target-rate-limit") a.target_rate_limit = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--preview-timeout-sec") a.preview_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--startup-sec") a.startup_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--startup-step-size") a.startup_step_size = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--startup-hold-sec") a.startup_hold_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--duration-sec") a.duration_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--cold-start-before-stand" || k == "--cold-start-before-policy") a.cold_start_before_stand = true;
    else if (k == "--cold-start-step-size") a.cold_start_step_size = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--cold-start-hold-sec") a.cold_start_hold_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--cold-start-kpkd-factor") a.cold_start_kpkd_factor = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--mode-machine") a.mode_machine = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--kp-factor") a.kp_factor = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--kd-factor") a.kd_factor = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--joint-pos-protect-ratio") a.joint_pos_protect_ratio = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--clip-by-torque") a.clip_by_torque = true;
    else if (k == "--no-clip-by-torque") a.clip_by_torque = false;
    else if (k == "--use-normalizer") a.use_normalizer = true;
    else if (k == "--no-use-normalizer") a.use_normalizer = false;
    else if (k == "--no-damping-on-stop") a.no_damping_on_stop = true;
    else if (k == "--experimental-no-prestart-hold") a.experimental_no_prestart_hold = true;
    else if (k == "--damping-on-stop") a.no_damping_on_stop = false;
    else if (k == "--check-only") a.check_only = true;
    else if (k == "--dry-run") a.dry_run = true;
    else if (k == "--dds-dry-run") a.dds_dry_run = true;
    else if (k == "--camera-check") a.camera_check = true;
    else if (k == "--depth-camera-check-only" || k == "--realsense-check-only") a.depth_camera_check_only = true;
    else if (k == "--require-depth-camera" || k == "--require-realsense") a.require_depth_camera = true;
    else if (k == "--depth-camera-timeout-sec" || k == "--realsense-timeout-sec") a.depth_camera_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--min-depth-camera-frames" || k == "--min-realsense-frames") a.min_depth_camera_frames = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--no-torso-imu") a.use_torso_imu = false;
    else if (k == "--require-torso-imu") { a.use_torso_imu = true; a.require_torso_imu = true; }
    else if (k == "--torso-imu-timeout-sec") a.torso_imu_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--stand-until-l1") a.stand_until_l1 = true;
    else if (k == "--stand-warmup-sec") a.stand_warmup_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--fail-on-bad-input") a.fail_on_bad_input = true;
    else if (k == "--max-preflight-tilt-deg") a.max_preflight_tilt_deg = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--max-preflight-dq") a.max_preflight_dq = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--max-preflight-action-abs") a.max_preflight_action_abs = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--max-preflight-target-err") a.max_preflight_target_err = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--min-preflight-rs-frames") a.min_preflight_rs_frames = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--stats-interval") a.stats_interval = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--lin-vel-deadband") a.lin_vel_deadband = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--lin-vel-range") a.lin_vel_range = parse_pair(i, argc, argv, k);
    else if (k == "--lin-vel-negative-range") a.lin_vel_negative_range = parse_pair(i, argc, argv, k);
    else if (k == "--lat-vel-range") a.lat_vel_range = parse_pair(i, argc, argv, k);
    else if (k == "--lat-vel-negative-range") a.lat_vel_negative_range = parse_pair(i, argc, argv, k);
    else if (k == "--ang-vel-deadband") a.ang_vel_deadband = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--ang-vel-range") a.ang_vel_range = parse_pair(i, argc, argv, k);
    else if (k == "--ang-vel-negative-range") a.ang_vel_negative_range = parse_pair(i, argc, argv, k);
    else if (k == "--help") {
      std::cout << "Usage:\n"
                << "  generic_sensor_policy_player --spec actions/instinct/config/generic_instinct_parkour.yaml [runtime overrides]\n"
                << "  generic_sensor_policy_player --spec actions/instinct/config/generic_instinct_parkour.yaml --depth-camera-check-only --require-depth-camera\n"
                << "  --check-only | --dry-run | --camera-check | --dds-dry-run | --depth-camera-check-only\n"
                << "  --depth-camera-check-only --require-depth-camera checks RealSense presence/frames without DDS/UserCtrl\n"
                << "  --require-torso-imu waits for rt/secondary_imu and uses torso IMU like official instinct_onboard\n"
                << "  --cold-start-before-policy runs the official ColdStartAgent-style transition before parkour/stand\n"
                << "  --standdir PATH --stand-until-l1 starts with the official stand policy; L1 switches to parkour\n"
                << "  --ready-file PATH --start-file PATH enables action_go prestart handoff to remove UserCtrl vacuum\n"
                << "  real run publishes to --cmd-topic rt/user_lowcmd\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }
  if (a.logdir.empty() && !a.depth_camera_check_only) throw std::runtime_error("--logdir is required");
  if (a.depth_camera_check_only) { a.depth_source = "realsense"; a.require_depth_camera = true; }
  if (a.torso_imu_topic.empty()) a.use_torso_imu = false;
  if (a.camera_check) { a.dry_run = true; a.depth_source = "realsense"; if (a.duration_sec <= 0) a.duration_sec = 5.0; if (a.stats_interval == 100) a.stats_interval = 25; }
  if (a.dds_dry_run) { if (a.duration_sec <= 0) a.duration_sec = 5.0; if (a.stats_interval == 100) a.stats_interval = 25; }
  if (!a.preview_file.empty() || !a.commit_file.empty() || !a.discard_file.empty()) {
    action_go::runtime::validate_preview_lifecycle(
        a.preview_file, a.commit_file, a.discard_file,
        a.preview_timeout_sec, a.cmd_topic);
  }
  return a;
}

std::array<float, 4> normalize_quat(std::array<float, 4> q) {
  float n = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  if (!(n > 1e-6f)) return {1.f,0.f,0.f,0.f};
  for (float& v : q) v /= n;
  return q;
}
std::array<float, 3> projected_gravity_from_quat(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w=q[0], x=q[1], y=q[2], z=q[3];
  const float r20 = 2.f * (x*z - y*w);
  const float r21 = 2.f * (y*z + x*w);
  const float r22 = 1.f - 2.f * (x*x + y*y);
  return {-r20, -r21, -r22};
}
float tilt_deg_from_quat(const std::array<float, 4>& q) {
  const auto g = projected_gravity_from_quat(q);
  const float c = std::clamp(-g[2], -1.f, 1.f);
  return std::acos(c) * 180.f / kPi;
}

float read_float_le(const std::array<uint8_t, 40>& bytes, size_t offset) {
  if (offset + 4 > bytes.size()) return 0.f;
  float v = 0.f;
  std::memcpy(&v, bytes.data() + offset, sizeof(float));
  if (!std::isfinite(v)) return 0.f;
  return std::clamp(v, -1.f, 1.f);
}

struct Command { float vx{0.f}, vy{0.f}, yaw{0.f}; };
uint16_t remote_keys(const std::array<uint8_t, 40>& remote) {
  return static_cast<uint16_t>(remote[2]) |
         (static_cast<uint16_t>(remote[3]) << 8);
}
bool remote_button(const std::array<uint8_t, 40>& remote, int button) {
  return ((remote_keys(remote) >> button) & 0x1) != 0;
}
Command joystick_command(const std::array<uint8_t, 40>& remote, const Args& args) {
  if (args.command_source == "zero") return {};
  const float lx_raw = read_float_le(remote, 4);
  const float rx_raw = read_float_le(remote, 8);
  const float ly_raw = read_float_le(remote, 20);
  Command c;
  const float ly = ly_raw;
  if (ly > args.lin_vel_deadband) {
    float a = (ly - args.lin_vel_deadband) / std::max(1e-6f, 1.f - args.lin_vel_deadband);
    c.vx = a * (args.lin_vel_range[1] - args.lin_vel_range[0]) + args.lin_vel_range[0];
  } else if (ly < -args.lin_vel_deadband) {
    float a = (ly + args.lin_vel_deadband) / std::max(1e-6f, 1.f - args.lin_vel_deadband);
    c.vx = a * (args.lin_vel_negative_range[1] - args.lin_vel_negative_range[0]) - args.lin_vel_negative_range[0];
  }
  const float lx = -lx_raw;
  if (lx > args.lin_vel_deadband) {
    float a = (lx - args.lin_vel_deadband) / std::max(1e-6f, 1.f - args.lin_vel_deadband);
    c.vy = a * (args.lat_vel_range[1] - args.lat_vel_range[0]) + args.lat_vel_range[0];
  } else if (lx < -args.lin_vel_deadband) {
    float a = (lx + args.lin_vel_deadband) / std::max(1e-6f, 1.f - args.lin_vel_deadband);
    c.vy = a * (args.lat_vel_negative_range[1] - args.lat_vel_negative_range[0]) - args.lat_vel_negative_range[0];
  }
  const float rx = -rx_raw;
  if (rx > args.ang_vel_deadband) {
    float a = (rx - args.ang_vel_deadband) / std::max(1e-6f, 1.f - args.ang_vel_deadband);
    c.yaw = a * (args.ang_vel_range[1] - args.ang_vel_range[0]) + args.ang_vel_range[0];
  } else if (rx < -args.ang_vel_deadband) {
    float a = (rx + args.ang_vel_deadband) / std::max(1e-6f, 1.f - args.ang_vel_deadband);
    c.yaw = a * (args.ang_vel_negative_range[1] - args.ang_vel_negative_range[0]) - args.ang_vel_negative_range[0];
  }
  return c;
}

std::string node_key(const YAML::Node& n) { return n.IsScalar() ? n.as<std::string>() : std::string{}; }
bool regex_match_joint(const std::string& pattern, const std::string& name) {
  try { return std::regex_search(name, std::regex(pattern)); }
  catch (...) { return false; }
}
std::vector<float> apply_regex_values(const YAML::Node& node, const std::vector<std::string>& joint_names, float def) {
  std::vector<float> out(joint_names.size(), def);
  if (!node || !node.IsMap()) return out;
  for (const auto& kv : node) {
    const std::string pat = node_key(kv.first);
    const float v = kv.second.as<float>();
    for (size_t i = 0; i < joint_names.size(); ++i) if (regex_match_joint(pat, joint_names[i])) out[i] = v;
  }
  return out;
}
std::vector<float> apply_regex_scale(const YAML::Node& node, const std::vector<std::string>& joint_names, float def) {
  if (!node) return std::vector<float>(kNumJoints, def);
  if (!node.IsMap()) return std::vector<float>(kNumJoints, node.as<float>());
  return apply_regex_values(node, joint_names, def);
}
float regex_value(const YAML::Node& node, const std::string& joint, float def) {
  if (!node) return def;
  if (!node.IsMap()) return node.as<float>();
  float out = def;
  for (const auto& kv : node) if (regex_match_joint(node_key(kv.first), joint)) out = kv.second.as<float>();
  return out;
}
std::vector<std::string> patterns_from_node(const YAML::Node& n) {
  std::vector<std::string> out;
  if (!n) return out;
  if (n.IsSequence()) for (const auto& v : n) out.push_back(v.as<std::string>());
  else out.push_back(n.as<std::string>());
  return out;
}
YAML::Node get_node(const YAML::Node& root, std::initializer_list<const char*> keys) {
  YAML::Node cur = root;
  for (auto* k : keys) {
    if (!cur || !cur.IsMap()) return YAML::Node();
    YAML::Node next = cur[k];
    if (!next || !next.IsDefined() || next.IsNull()) return YAML::Node();
    cur = next;
  }
  return cur;
}
bool valid_scalar(const YAML::Node& n) { return n && !n.IsNull() && n.IsScalar(); }
int as_int_or(const YAML::Node& n, int def) {
  if (!valid_scalar(n)) return def;
  try { return n.as<int>(); } catch (...) { return def; }
}
float as_float_or(const YAML::Node& n, float def) {
  if (!valid_scalar(n)) return def;
  try { return n.as<float>(); } catch (...) { return def; }
}
double as_double_or(const YAML::Node& n, double def) {
  if (!valid_scalar(n)) return def;
  try { return n.as<double>(); } catch (...) { return def; }
}
bool as_bool_or(const YAML::Node& n, bool def) {
  if (!valid_scalar(n)) return def;
  try { return n.as<bool>(); } catch (...) { return def; }
}

class HistoryBuffer {
 public:
  explicit HistoryBuffer(int length=0) : length_(length) {}
  std::vector<float> append(const std::vector<float>& value) {
    if (length_ <= 0) return value;
    const int dim = static_cast<int>(value.size());
    if (buffer_.empty()) buffer_.assign(static_cast<size_t>(length_ * dim), 0.f);
    if (pushes_ == 0) {
      for (int i = 0; i < length_; ++i) std::copy(value.begin(), value.end(), buffer_.begin() + i * dim);
    } else {
      std::move(buffer_.begin() + dim, buffer_.end(), buffer_.begin());
      std::copy(value.begin(), value.end(), buffer_.end() - dim);
    }
    ++pushes_;
    return buffer_;
  }
 private:
  int length_{0};
  int pushes_{0};
  std::vector<float> buffer_;
};

struct RobotState {
  bool connected{false};
  bool torso_imu_connected{false};
  int mode_machine{0};
  uint64_t seq{0};
  std::chrono::steady_clock::time_point recv_stamp{};
  std::array<float, 4> quat{1.f,0.f,0.f,0.f};
  std::array<float, 3> gyro{0.f,0.f,0.f};
  std::vector<float> q_sim = std::vector<float>(kNumJoints, 0.f);
  std::vector<float> dq_sim = std::vector<float>(kNumJoints, 0.f);
  std::array<uint8_t, 40> wireless_remote{};
  double torso_imu_age_ms{std::numeric_limits<double>::infinity()};
  const char* imu_source{"pelvis"};
};

struct DepthConfig {
  int out_w{64}, out_h{36};
  std::array<int,4> crop{18,0,16,16};
  bool has_crop{true};
  float range_lo{0.f}, range_hi{2.5f};
  float output_lo{0.f}, output_hi{1.f};
  int kernel{3};
  double sigma{1.0};
  int history_skip_frames{5};
  int num_frames{8};
  int depth_h{18};
  int depth_w{32};
  float sim_update_period{0.02f};
  int camera_history_length{37};
};

class DepthSource {
 public:
  virtual ~DepthSource() = default;
  virtual std::vector<float> observation() = 0;
  virtual std::string stats() const { return {}; }
  virtual int frame_count() const { return 0; }
  virtual uint64_t sequence() const { return static_cast<uint64_t>(std::max(0, frame_count())); }
  virtual double age_sec() const { return 0.0; }
};

class ZeroDepthSource final : public DepthSource {
 public:
  explicit ZeroDepthSource(const DepthConfig& cfg) : cfg_(cfg), obs_(static_cast<size_t>(cfg.num_frames * cfg.depth_h * cfg.depth_w), 0.f) {}
  std::vector<float> observation() override { return obs_; }
  std::string stats() const override { return "source=zero"; }
  int frame_count() const override { return 1; }
  uint64_t sequence() const override { return 1; }
 private:
  DepthConfig cfg_;
  std::vector<float> obs_;
};

class RealSenseDepthSource final : public DepthSource {
 public:
  RealSenseDepthSource(const DepthConfig& cfg, const Args& args) : cfg_(cfg), args_(args) {
#ifndef ACTION_GO_WITH_REALSENSE
    (void)cfg_; (void)args_;
    throw std::runtime_error("generic_sensor_policy_player instinct_parkour variant was built without librealsense2; use --depth-source zero or install librealsense2-dev and rebuild");
#else
    if (args_.require_depth_camera) {
      rs2::context ctx;
      const auto devices = ctx.query_devices();
      if (devices.size() == 0) {
        throw std::runtime_error("no RealSense depth camera detected");
      }
    }
    rs2::config c;
    c.enable_stream(RS2_STREAM_DEPTH, args_.rs_width, args_.rs_height, RS2_FORMAT_Z16, args_.rs_fps);
    profile_ = pipe_.start(c);
    depth_scale_ = profile_.get_device().first<rs2::depth_sensor>().get_depth_scale();
    (void)pipe_.wait_for_frames(1000);
    const int sim_frequency = std::max(1, static_cast<int>(1.0f / std::max(1e-6f, cfg_.sim_update_period)));
    real_skip_ = std::max(1, static_cast<int>(args_.rs_fps / static_cast<float>(sim_frequency) * cfg_.history_skip_frames));
    const int first = -1 - real_skip_ * (cfg_.num_frames - 1);
    indices_.clear();
    for (int i = 0; i < cfg_.num_frames; ++i) {
      const double a = cfg_.num_frames == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(cfg_.num_frames - 1);
      indices_.push_back(static_cast<int>(std::lround(first * (1.0 - a) + (-1) * a)));
    }
    buffer_len_ = std::max(args_.rs_fps, std::abs(indices_.front()) + 2);
    stop_.store(false);
    thread_ = std::thread(&RealSenseDepthSource::thread_main, this);
    const auto timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(std::max(0.0, args_.depth_camera_timeout_sec)));
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (static_cast<int>(frame_count_.load()) >= std::max(1, args_.min_depth_camera_frames) ||
            !frames_.empty()) {
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (args_.require_depth_camera &&
        frame_count_.load() < std::max(1, args_.min_depth_camera_frames)) {
      stop_.store(true);
      if (thread_.joinable()) thread_.join();
      try { pipe_.stop(); } catch (...) {}
      throw std::runtime_error("RealSense depth camera produced no frames within " +
                               std::to_string(args_.depth_camera_timeout_sec) + " sec");
    }
    std::cout << "[generic_sensor_policy_player][instinct_parkour] RealSense started resolution=(" << args_.rs_width << ", " << args_.rs_height
              << ") fps=" << args_.rs_fps << " sim_skip=" << cfg_.history_skip_frames << " real_skip=" << real_skip_ << " depth_indices=[";
    for (size_t i = 0; i < indices_.size(); ++i) std::cout << (i ? "," : "") << indices_[i];
    std::cout << "]\n";
#endif
  }
  ~RealSenseDepthSource() override {
#ifdef ACTION_GO_WITH_REALSENSE
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    try { pipe_.stop(); } catch (...) {}
#endif
  }

  std::vector<float> observation() override {
    std::lock_guard<std::mutex> lk(mtx_);
    if (frames_.empty()) return std::vector<float>(static_cast<size_t>(cfg_.num_frames * cfg_.depth_h * cfg_.depth_w), 0.f);
    std::vector<float> out;
    out.reserve(static_cast<size_t>(cfg_.num_frames * cfg_.depth_h * cfg_.depth_w));
    for (int idx : indices_) {
      int pos = static_cast<int>(frames_.size()) + idx;
      pos = std::clamp(pos, 0, static_cast<int>(frames_.size()) - 1);
      out.insert(out.end(), frames_[pos].begin(), frames_[pos].end());
    }
    return out;
  }
  int frame_count() const override { return frame_count_.load(); }
  double age_sec() const override {
    std::lock_guard<std::mutex> lk(mtx_);
    if (last_time_.time_since_epoch().count() == 0) {
      return std::numeric_limits<double>::infinity();
    }
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - last_time_).count();
  }
  std::string stats() const override {
    std::lock_guard<std::mutex> lk(mtx_);
    const double age_ms = last_time_.time_since_epoch().count() == 0 ? std::numeric_limits<double>::infinity() :
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last_time_).count();
    std::ostringstream ss;
    ss << "rs_frames=" << frame_count_.load() << " buffered=" << frames_.size() << " age=" << std::fixed << std::setprecision(0) << age_ms << "ms "
       << std::setprecision(2) << "raw_m=[" << raw_min_ << "," << raw_max_ << "] raw_mean=" << raw_mean_ << " zero=" << zero_frac_ << " "
       << "proc=[" << proc_min_ << "," << proc_max_ << "] proc_mean=" << proc_mean_;
    return ss.str();
  }
 private:
  std::vector<float> preprocess(const cv::Mat& depth_m) {
    cv::Mat depth, cropped;
    cv::resize(depth_m, depth, cv::Size(cfg_.out_w, cfg_.out_h), 0, 0, cv::INTER_NEAREST);
    if (cfg_.has_crop) {
      const int x1=cfg_.crop[0], x2=cfg_.crop[1], y1=cfg_.crop[2], y2=cfg_.crop[3];
      cropped = depth(cv::Range(x1, depth.rows - x2), cv::Range(y1, depth.cols - y2)).clone();
    } else cropped = depth;
    cv::Mat mask = cropped < 0.2f;
    if (cv::countNonZero(mask) > 0) cv::inpaint(cropped, mask, cropped, 3.0, cv::INPAINT_NS);
    if (cfg_.kernel > 0) cv::GaussianBlur(cropped, cropped, cv::Size(cfg_.kernel, cfg_.kernel), cfg_.sigma, cfg_.sigma);
    std::vector<float> out(static_cast<size_t>(cropped.rows * cropped.cols));
    float mn=1e9f, mx=-1e9f, sum=0.f;
    for (int r=0; r<cropped.rows; ++r) for (int c=0; c<cropped.cols; ++c) {
      float v = cropped.at<float>(r,c);
      v = std::clamp(v, cfg_.range_lo, cfg_.range_hi);
      float norm = (v - cfg_.range_lo) / std::max(1e-6f, cfg_.range_hi - cfg_.range_lo);
      float o = norm * (cfg_.output_hi - cfg_.output_lo) + cfg_.output_lo;
      out[static_cast<size_t>(r*cropped.cols+c)] = o;
      mn = std::min(mn, o); mx = std::max(mx, o); sum += o;
    }
    proc_min_ = mn; proc_max_ = mx; proc_mean_ = sum / std::max<size_t>(1, out.size());
    return out;
  }
#ifdef ACTION_GO_WITH_REALSENSE
  void thread_main() {
    while (!stop_.load() && !g_stop_requested.load()) {
      try {
        auto frameset = pipe_.wait_for_frames(std::max(1, int(2000 / std::max(1, args_.rs_fps))));
        auto df = frameset.get_depth_frame();
        if (!df) continue;
        cv::Mat raw(cv::Size(args_.rs_width, args_.rs_height), CV_16UC1, (void*)df.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_m;
        raw.convertTo(depth_m, CV_32FC1, depth_scale_);
        int valid_count = 0; float mn=1e9f, mx=-1e9f, sum=0.f;
        for (int r=0; r<depth_m.rows; ++r) for (int c=0; c<depth_m.cols; ++c) {
          float v = depth_m.at<float>(r,c);
          if (v > 0.f) { ++valid_count; mn=std::min(mn,v); mx=std::max(mx,v); sum += v; }
        }
        if (valid_count > 0) { raw_min_=mn; raw_max_=mx; raw_mean_=sum/valid_count; zero_frac_=1.f - valid_count / static_cast<float>(depth_m.rows*depth_m.cols); }
        else { raw_min_=raw_max_=raw_mean_=0.f; zero_frac_=1.f; }
        auto frame = preprocess(depth_m);
        std::lock_guard<std::mutex> lk(mtx_);
        ++frame_count_;
        last_time_ = std::chrono::steady_clock::now();
        if (frames_.empty()) for (int i=0;i<buffer_len_;++i) frames_.push_back(frame);
        else { if ((int)frames_.size() >= buffer_len_) frames_.erase(frames_.begin()); frames_.push_back(std::move(frame)); }
      } catch (const std::exception& e) {
        std::cerr << "[generic_sensor_policy_player][instinct_parkour] RealSense warning: " << e.what() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  }
  rs2::pipeline pipe_;
  rs2::pipeline_profile profile_;
  float depth_scale_{0.001f};
  std::thread thread_;
  std::atomic<bool> stop_{false};
#endif
  DepthConfig cfg_;
  Args args_;
  int real_skip_{5};
  int buffer_len_{60};
  std::vector<int> indices_{-36,-31,-26,-21,-16,-11,-6,-1};
  mutable std::mutex mtx_;
  std::vector<std::vector<float>> frames_;
  std::atomic<int> frame_count_{0};
  std::chrono::steady_clock::time_point last_time_{};
  mutable float raw_min_{0.f}, raw_max_{0.f}, raw_mean_{0.f}, zero_frac_{1.f};
  mutable float proc_min_{0.f}, proc_max_{0.f}, proc_mean_{0.f};
};

int run_depth_camera_check_only(const Args& args) {
  DepthConfig cfg;
  std::unique_ptr<DepthSource> depth = std::make_unique<RealSenseDepthSource>(cfg, args);
  std::cout << "[generic_sensor_policy_player][instinct_parkour] depth-camera-check OK: "
            << depth->stats() << "\n";
  return 0;
}

void touch_file(const std::string& path) {
  if (path.empty()) return;
  const fs::path p(path);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream out(path);
  if (!out.good()) throw std::runtime_error("failed to touch file: " + path);
  out << "ready\n";
}

std::string hash_float_vector(const std::vector<float>& values) {
  uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  for (size_t i = 0; i < values.size() * sizeof(float); ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

class InstinctPolicy {
 public:
  struct StateSnapshot {
    std::map<std::string, HistoryBuffer> histories;
    std::vector<float> last_action;
  };

  explicit InstinctPolicy(Args args) : args_(std::move(args)), logdir_(fs::absolute(args_.logdir)) {
    // New action_go layout uses model/ + config/.  Keep compatibility with
    // older exported training directories that used exported/ + params/.
    exported_ = fs::exists(logdir_/"model") ? (logdir_/"model") : (logdir_/"exported");
    params_ = fs::exists(logdir_/"config") ? (logdir_/"config") : (logdir_/"params");
    if (!fs::exists(exported_/"0-depth_encoder.onnx") || !fs::exists(exported_/"actor.onnx")) throw std::runtime_error("missing ONNX files under " + exported_.string());
    env_cfg_ = YAML::LoadFile((params_/"env.yaml").string());
    parse_config();
    load_models();
    for (const auto& kv : history_len_) histories_.emplace(kv.first, HistoryBuffer(kv.second));
    last_action_.assign(kNumJoints, 0.f);
  }
  const DepthConfig& depth_config() const { return depth_cfg_; }
  const std::vector<float>& default_joint_pos() const { return default_joint_pos_; }
  const std::vector<float>& kp() const { return kp_; }
  const std::vector<float>& kd() const { return kd_; }
  const std::vector<std::string>& joint_names() const { return joint_names_; }
  const std::vector<int>& joint_map() const { return joint_map_; }
  const std::vector<float>& joint_signs() const { return joint_signs_; }
  const std::vector<float>& lower_limits() const { return lower_limits_; }
  const std::vector<float>& upper_limits() const { return upper_limits_; }
  const std::string& joint_name(int i) const {
    static const std::string unknown = "unknown";
    return i >= 0 && i < static_cast<int>(joint_names_.size()) ? joint_names_[static_cast<size_t>(i)] : unknown;
  }
  void seed_last_action(const std::vector<float>& action) {
    if (action.size() == static_cast<size_t>(kNumJoints)) last_action_ = action;
  }
  StateSnapshot snapshot_state() const { return {histories_, last_action_}; }
  void restore_state(const StateSnapshot& snapshot) {
    histories_ = snapshot.histories;
    last_action_ = snapshot.last_action;
  }
  std::vector<float> target_to_action(const std::vector<float>& target) const {
    std::vector<float> action(kNumJoints, 0.f);
    for (int i = 0; i < kNumJoints; ++i) {
      const float scale = std::abs(action_scale_[static_cast<size_t>(i)]) > 1e-6f
          ? action_scale_[static_cast<size_t>(i)] : 1.f;
      action[static_cast<size_t>(i)] =
          (target[static_cast<size_t>(i)] - action_offset_[static_cast<size_t>(i)]) / scale;
    }
    return action;
  }
  void check() const {
    std::cout << "[generic_sensor_policy_player][instinct_parkour] check-only OK\n"
              << "  logdir: " << logdir_ << "\n"
              << "  raw_obs_dim=" << raw_dim_ << " proprio_dim=" << proprio_dim_ << " normalizer_source=none use_normalizer=" << (args_.use_normalizer?"true":"false") << "\n"
              << "  depth_encoder input=[" << depth_shape_[0] << ", " << depth_shape_[1] << ", " << depth_shape_[2] << ", " << depth_shape_[3] << "] output=[" << depth_out_shape_[0] << ", " << depth_out_shape_[1] << "]\n"
              << "  actor input=[" << actor_shape_[0] << ", " << actor_shape_[1] << "] output=[" << actor_out_shape_[0] << ", " << actor_out_shape_[1] << "]\n"
              << "  joint_order first_joints=["
              << joint_name(0) << ", " << joint_name(1) << ", " << joint_name(2) << "]\n"
              << "  default_joint_pos: min=" << *std::min_element(default_joint_pos_.begin(), default_joint_pos_.end()) << " max=" << *std::max_element(default_joint_pos_.begin(), default_joint_pos_.end()) << "\n"
              << "  action_scale: min=" << *std::min_element(action_scale_.begin(), action_scale_.end()) << " max=" << *std::max_element(action_scale_.begin(), action_scale_.end()) << "\n"
              << "  kp: min=" << *std::min_element(kp_.begin(), kp_.end()) << " max=" << *std::max_element(kp_.begin(), kp_.end()) << "; kd: min=" << *std::min_element(kd_.begin(), kd_.end()) << " max=" << *std::max_element(kd_.begin(), kd_.end()) << "\n"
              << "  depth_cfg: output_resolution=(" << depth_cfg_.out_w << "," << depth_cfg_.out_h << ") crop=[" << depth_cfg_.crop[0] << "," << depth_cfg_.crop[1] << "," << depth_cfg_.crop[2] << "," << depth_cfg_.crop[3] << "] depth_shape=(" << depth_cfg_.num_frames << "," << depth_cfg_.depth_h << "," << depth_cfg_.depth_w << ")\n";
  }
  std::vector<float> infer(const RobotState& state, const Command& command, const std::vector<float>& depth_obs) {
    auto raw = build_raw_obs(state, command, depth_obs);
    const int depth_size = depth_cfg_.num_frames * depth_cfg_.depth_h * depth_cfg_.depth_w;
    if (raw.size() != static_cast<size_t>(raw_dim_)) {
      throw std::runtime_error("obs dim mismatch got=" + std::to_string(raw.size()) +
                               " expected=" + std::to_string(raw_dim_) +
                               " depth_obs=" + std::to_string(depth_obs.size()) +
                               " depth_expected=" + std::to_string(depth_size) +
                               " proprio=" + std::to_string(proprio_dim_));
    }
    const int latent_dim = depth_out_shape_[1];
    std::vector<float> depth_batch(static_cast<size_t>(batch_ * depth_size));
    for (int b=0; b<batch_; ++b) std::copy(raw.end()-depth_size, raw.end(), depth_batch.begin()+b*depth_size);
    std::array<int64_t,4> dshape{batch_, depth_cfg_.num_frames, depth_cfg_.depth_h, depth_cfg_.depth_w};
    Ort::Value dtensor = Ort::Value::CreateTensor<float>(mem_, depth_batch.data(), depth_batch.size(), dshape.data(), dshape.size());
    const char* din = depth_input_name_.c_str(); const char* dout = depth_output_name_.c_str();
    auto douts = depth_sess_->Run(Ort::RunOptions{nullptr}, &din, &dtensor, 1, &dout, 1);
    const auto latent_vec = tensor_to_vec_exact(douts[0], static_cast<size_t>(batch_ * latent_dim), "Instinct depth encoder");
    const float* latent = latent_vec.data();
    std::vector<float> actor_in(static_cast<size_t>(batch_ * actor_shape_[1]));
    for (int b=0; b<batch_; ++b) {
      std::copy(raw.begin(), raw.end()-depth_size, actor_in.begin()+b*actor_shape_[1]);
      std::copy(latent+b*latent_dim, latent+(b+1)*latent_dim, actor_in.begin()+b*actor_shape_[1]+proprio_dim_);
    }
    std::array<int64_t,2> ashape{batch_, actor_shape_[1]};
    Ort::Value atensor = Ort::Value::CreateTensor<float>(mem_, actor_in.data(), actor_in.size(), ashape.data(), ashape.size());
    const char* ain = actor_input_name_.c_str(); const char* aout = actor_output_name_.c_str();
    auto aouts = actor_sess_->Run(Ort::RunOptions{nullptr}, &ain, &atensor, 1, &aout, 1);
    const auto action_vec = tensor_to_vec_exact(aouts[0], static_cast<size_t>(actor_out_shape_[0] * actor_out_shape_[1]), "Instinct actor");
    const float* act = action_vec.data();
    std::vector<float> action(act, act + kNumJoints);
    last_action_ = action;
    return action;
  }
  std::vector<float> action_to_target(const std::vector<float>& action,
                                      const RobotState& state,
                                      const std::vector<float>* kp_override = nullptr,
                                      const std::vector<float>* kd_override = nullptr) const {
    std::vector<float> target(kNumJoints);
    for (int i=0; i<kNumJoints; ++i) {
      target[i] = action_offset_[i] + action[i] * action_scale_[i];
      target[i] = std::clamp(target[i], lower_limits_[static_cast<size_t>(i)], upper_limits_[static_cast<size_t>(i)]);
      if (args_.clip_by_torque) {
        const float kp = kp_override && kp_override->size() == static_cast<size_t>(kNumJoints)
            ? (*kp_override)[static_cast<size_t>(i)] : kp_[static_cast<size_t>(i)];
        const float kd = kd_override && kd_override->size() == static_cast<size_t>(kNumJoints)
            ? (*kd_override)[static_cast<size_t>(i)] : kd_[static_cast<size_t>(i)];
        const float p = std::max(kp, 1e-6f);
        const float lo = (-torque_limits_[static_cast<size_t>(i)] + kd * state.dq_sim[i]) / p + state.q_sim[i];
        const float hi = ( torque_limits_[static_cast<size_t>(i)] + kd * state.dq_sim[i]) / p + state.q_sim[i];
        target[i] = std::clamp(target[i], lo, hi);
      }
    }
    return target;
  }
 private:
  void parse_config() {
    yamlc::require_size("robot.motor_count", kNumJoints,
                        static_cast<size_t>(yamlc::required_scalar<int>(env_cfg_, "robot.motor_count")));
    joint_names_ = yamlc::required_vector_size<std::string>(env_cfg_, "robot.policy_joint_order", kNumJoints);
    joint_map_ = yamlc::required_vector_size<int>(env_cfg_, "robot.joint_map", kNumJoints);
    yamlc::require_permutation("robot.joint_map", joint_map_, kNumJoints);
    joint_signs_ = yamlc::required_vector_size<float>(env_cfg_, "robot.joint_signs", kNumJoints);
    default_joint_pos_ = yamlc::required_vector_size<float>(env_cfg_, "robot.default_angles", kNumJoints);
    default_joint_vel_ = yamlc::required_vector_size<float>(env_cfg_, "robot.default_velocities", kNumJoints);
    action_scale_ = yamlc::required_vector_size<float>(env_cfg_, "robot.action_scale", kNumJoints);
    action_offset_ = yamlc::required_vector_size<float>(env_cfg_, "robot.action_offset", kNumJoints);
    kp_ = yamlc::required_vector_size<float>(env_cfg_, "robot.kp", kNumJoints);
    kd_ = yamlc::required_vector_size<float>(env_cfg_, "robot.kd", kNumJoints);
    torque_limits_ = yamlc::required_vector_size<float>(env_cfg_, "robot.torque_limits", kNumJoints);
    lower_limits_ = yamlc::required_vector_size<float>(env_cfg_, "robot.joint_pos_lower_limit", kNumJoints);
    upper_limits_ = yamlc::required_vector_size<float>(env_cfg_, "robot.joint_pos_upper_limit", kNumJoints);
    for (int i=0; i<kNumJoints; ++i) { kp_[i] = std::clamp(kp_[i] * args_.kp_factor, 0.f, 500.f); kd_[i] = std::clamp(kd_[i] * args_.kd_factor, 0.f, 5.f); }
    parse_observation_terms();
    parse_camera();
    raw_dim_ = 0; for (const auto& kv : policy_format_) raw_dim_ += kv.second;
    const int depth_size = depth_cfg_.num_frames * depth_cfg_.depth_h * depth_cfg_.depth_w;
    proprio_dim_ = raw_dim_ - depth_size;
    if (args_.use_normalizer) throw std::runtime_error("new parkour C++ path supports this onboard export without policy_normalizer; use --no-use-normalizer");
  }
  void parse_actuator(const YAML::Node& actuator) {
    YAML::Node cfg = actuator["base_cfg"] ? actuator["base_cfg"] : actuator;
    YAML::Node names = cfg["target_names_expr"] ? cfg["target_names_expr"] : (cfg["joint_names_expr"] ? cfg["joint_names_expr"] : actuator["joint_names_expr"]);
    auto patterns = patterns_from_node(names);
    for (int i=0; i<kNumJoints; ++i) {
      bool matched=false; for (const auto& p : patterns) if (regex_match_joint(p, joint_names_[static_cast<size_t>(i)])) { matched=true; break; }
      if (matched) {
        kp_[i] = regex_value(cfg["stiffness"], joint_names_[static_cast<size_t>(i)], kp_[i]);
        kd_[i] = regex_value(cfg["damping"], joint_names_[static_cast<size_t>(i)], kd_[i]);
      }
    }
  }
  void parse_observation_terms() {
    YAML::Node policy = env_cfg_["observations"] ? env_cfg_["observations"]["policy"] : YAML::Node();
    YAML::Node terms = policy && policy["terms"] ? policy["terms"] : policy;
    if (!terms || !terms.IsMap() || terms.size() == 0) {
      // Deployment fallback for the current onboard export.  The model's actor
      // input is 768 proprio + 128 depth latent; proprio is the seven policy
      // terms below with 8-frame history except depth.
      obs_scales_["base_ang_vel"] = 0.25f;
      obs_scales_["joint_vel"] = 0.05f;
      for (const std::string& name : {"base_ang_vel","projected_gravity","velocity_commands"}) {
        history_len_[name] = 8;
        policy_format_.push_back({name, 24});
      }
      for (const std::string& name : {"joint_pos","joint_vel","actions"}) {
        history_len_[name] = 8;
        policy_format_.push_back({name, 232});
      }
      policy_format_.push_back({"depth_image", 0});
      return;
    }
    const std::vector<std::string> order{"base_ang_vel","projected_gravity","velocity_commands","joint_pos","joint_vel","actions","depth_image"};
    for (const auto& name : order) {
      YAML::Node cfg = terms[name];
      if (!cfg) continue;
      if (valid_scalar(cfg["scale"])) obs_scales_[name] = as_float_or(cfg["scale"], 1.f);
      int hist = as_int_or(cfg["history_length"], 0);
      if (hist > 0) history_len_[name] = hist;
      if (name == "depth_image") policy_format_.push_back({name, 0});
      else if (name == "base_ang_vel" || name == "projected_gravity" || name == "velocity_commands") policy_format_.push_back({name, 3 * std::max(1, hist)});
      else policy_format_.push_back({name, kNumJoints * std::max(1, hist)});
    }
    depth_params_ = terms["depth_image"] ? terms["depth_image"]["params"] : YAML::Node();
  }
  void parse_camera() {
    YAML::Node scene = env_cfg_["scene"];
    YAML::Node camera = scene["camera"];
    if (!camera) {
      YAML::Node sensors = scene["sensors"];
      if (sensors && sensors.IsSequence()) {
        for (auto it = sensors.begin(); it != sensors.end(); ++it) {
          YAML::Node sensor = *it;
          if (sensor["name"] && sensor["name"].as<std::string>() == "camera") { camera = sensor; break; }
        }
      }
    }
    if (!camera) throw std::runtime_error("camera config not found in env.yaml");
    YAML::Node pattern = camera["pattern"] ? camera["pattern"] : camera["pattern_cfg"];
    depth_cfg_.out_w = as_int_or(pattern["width"], 64);
    depth_cfg_.out_h = as_int_or(pattern["height"], 36);
    YAML::Node noise = camera["noise_pipeline"];
    YAML::Node crop = get_node(noise, {"crop_and_resize","crop_region"});
    if (crop && crop.IsSequence()) { depth_cfg_.has_crop = true; for (int i=0;i<4;++i) depth_cfg_.crop[i]=as_int_or(crop[i], depth_cfg_.crop[i]); }
    YAML::Node norm = noise["depth_normalization"];
    if (norm["depth_range"]) { depth_cfg_.range_lo = as_float_or(norm["depth_range"][0], depth_cfg_.range_lo); depth_cfg_.range_hi = as_float_or(norm["depth_range"][1], depth_cfg_.range_hi); }
    if (norm["output_range"]) { depth_cfg_.output_lo = as_float_or(norm["output_range"][0], depth_cfg_.output_lo); depth_cfg_.output_hi = as_float_or(norm["output_range"][1], depth_cfg_.output_hi); }
    YAML::Node blur = noise["gaussian_blur"];
    if (blur["kernel_size"]) depth_cfg_.kernel = as_int_or(blur["kernel_size"], depth_cfg_.kernel);
    if (blur["sigma"]) depth_cfg_.sigma = as_double_or(blur["sigma"], depth_cfg_.sigma);
    if (camera["update_period"]) depth_cfg_.sim_update_period = as_float_or(camera["update_period"], depth_cfg_.sim_update_period);
    if (depth_params_["history_skip_frames"]) depth_cfg_.history_skip_frames = as_int_or(depth_params_["history_skip_frames"], depth_cfg_.history_skip_frames);
    if (depth_params_["time_downsample_factor"]) depth_cfg_.history_skip_frames = as_int_or(depth_params_["time_downsample_factor"], depth_cfg_.history_skip_frames);
    YAML::Node hist = get_node(camera, {"data_histories","distance_to_image_plane_noised"});
    // The deployed policy/ONNX depth encoder is exported for the policy
    // observation term's num_output_frames.  Some training env.yaml files keep a
    // larger camera data_history only as a source buffer.  Prefer
    // num_output_frames here; otherwise RealSense can produce e.g. 16 frames
    // while the actor raw observation expects 8, causing "obs dim mismatch".
    if (depth_params_["num_output_frames"]) {
      depth_cfg_.num_frames = as_int_or(depth_params_["num_output_frames"], depth_cfg_.num_frames);
    } else if (hist) {
      depth_cfg_.camera_history_length = as_int_or(hist, depth_cfg_.camera_history_length);
      depth_cfg_.num_frames = (depth_cfg_.camera_history_length - 1) / std::max(1, depth_cfg_.history_skip_frames) + 1;
    }
    if (depth_cfg_.has_crop) { depth_cfg_.depth_h = depth_cfg_.out_h - depth_cfg_.crop[0] - depth_cfg_.crop[1]; depth_cfg_.depth_w = depth_cfg_.out_w - depth_cfg_.crop[2] - depth_cfg_.crop[3]; }
    else { depth_cfg_.depth_h = depth_cfg_.out_h; depth_cfg_.depth_w = depth_cfg_.out_w; }
    for (auto& kv : policy_format_) if (kv.first == "depth_image") kv.second = depth_cfg_.num_frames * depth_cfg_.depth_h * depth_cfg_.depth_w;
  }
  static int64_t dim_or(const std::vector<int64_t>& s, size_t i, int64_t def) { return i < s.size() && s[i] > 0 ? s[i] : def; }
  void load_models() {
    Ort::SessionOptions opt; opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); opt.SetIntraOpNumThreads(std::max(1, args_.onnx_threads)); opt.SetInterOpNumThreads(1);
    depth_sess_ = std::make_unique<Ort::Session>(env_, (exported_/"0-depth_encoder.onnx").c_str(), opt);
    actor_sess_ = std::make_unique<Ort::Session>(env_, (exported_/"actor.onnx").c_str(), opt);
    Ort::AllocatorWithDefaultOptions alloc;
    depth_input_name_ = depth_sess_->GetInputNameAllocated(0, alloc).get(); depth_output_name_ = depth_sess_->GetOutputNameAllocated(0, alloc).get();
    actor_input_name_ = actor_sess_->GetInputNameAllocated(0, alloc).get(); actor_output_name_ = actor_sess_->GetOutputNameAllocated(0, alloc).get();
    auto ds = depth_sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto dos = depth_sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto as = actor_sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto aos = actor_sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    depth_shape_ = {dim_or(ds,0,1), dim_or(ds,1,depth_cfg_.num_frames), dim_or(ds,2,depth_cfg_.depth_h), dim_or(ds,3,depth_cfg_.depth_w)};
    depth_out_shape_ = {dim_or(dos,0,depth_shape_[0]), dim_or(dos,1,128)};
    actor_shape_ = {dim_or(as,0,depth_shape_[0]), dim_or(as,1,proprio_dim_+depth_out_shape_[1])};
    actor_out_shape_ = {dim_or(aos,0,actor_shape_[0]), dim_or(aos,1,kNumJoints)};
    batch_ = static_cast<int>(actor_shape_[0]);
    if (depth_shape_[1] != depth_cfg_.num_frames || depth_shape_[2] != depth_cfg_.depth_h || depth_shape_[3] != depth_cfg_.depth_w) throw std::runtime_error("depth encoder shape does not match env depth config");
    if (actor_shape_[0] != depth_shape_[0]) throw std::runtime_error("actor/depth batch mismatch");
    if (actor_shape_[1] != proprio_dim_ + depth_out_shape_[1]) {
      throw std::runtime_error("actor input dim mismatch actor=" + std::to_string(actor_shape_[1]) +
                               " proprio=" + std::to_string(proprio_dim_) +
                               " latent=" + std::to_string(depth_out_shape_[1]) +
                               " raw=" + std::to_string(raw_dim_) +
                               " depth_size=" + std::to_string(depth_cfg_.num_frames * depth_cfg_.depth_h * depth_cfg_.depth_w));
    }
    if (actor_out_shape_[1] != kNumJoints) throw std::runtime_error("actor output dim mismatch");
  }
  std::vector<float> hist_term(const std::string& name, const std::vector<float>& v) {
    auto it = histories_.find(name); if (it == histories_.end()) return v; return it->second.append(v);
  }
  std::vector<float> build_raw_obs(const RobotState& s, const Command& c, const std::vector<float>& depth) {
    std::map<std::string, std::vector<float>> terms;
    terms["base_ang_vel"] = {s.gyro[0] * scale("base_ang_vel",1.f), s.gyro[1] * scale("base_ang_vel",1.f), s.gyro[2] * scale("base_ang_vel",1.f)};
    const auto g = projected_gravity_from_quat(s.quat); terms["projected_gravity"] = {g[0],g[1],g[2]};
    terms["velocity_commands"] = {c.vx,c.vy,c.yaw};
    terms["joint_pos"].resize(kNumJoints); terms["joint_vel"].resize(kNumJoints);
    for (int i=0;i<kNumJoints;++i) { terms["joint_pos"][i] = s.q_sim[i] - default_joint_pos_[i]; terms["joint_vel"][i] = (s.dq_sim[i] - default_joint_vel_[i]) * scale("joint_vel",1.f); }
    terms["actions"] = last_action_;
    std::vector<float> obs; obs.reserve(raw_dim_);
    for (const auto& kv : policy_format_) {
      if (kv.first == "depth_image") obs.insert(obs.end(), depth.begin(), depth.end());
      else { auto h = hist_term(kv.first, terms[kv.first]); obs.insert(obs.end(), h.begin(), h.end()); }
    }
    return obs;
  }
  float scale(const std::string& k, float def) const { auto it=obs_scales_.find(k); return it==obs_scales_.end()?def:it->second; }

  Args args_;
  fs::path logdir_, exported_, params_;
  YAML::Node env_cfg_, depth_params_;
  DepthConfig depth_cfg_;
  std::vector<float> default_joint_pos_, default_joint_vel_, action_scale_, action_offset_, kp_, kd_, last_action_;
  std::vector<float> torque_limits_, lower_limits_, upper_limits_, joint_signs_;
  std::vector<int> joint_map_;
  std::vector<std::string> joint_names_;
  std::vector<std::pair<std::string,int>> policy_format_;
  std::map<std::string,float> obs_scales_;
  std::map<std::string,int> history_len_;
  std::map<std::string,HistoryBuffer> histories_;
  int raw_dim_{0}, proprio_dim_{0}, batch_{1};
  std::array<int64_t,4> depth_shape_{1,8,18,32};
  std::array<int64_t,2> depth_out_shape_{1,128}, actor_shape_{1,896}, actor_out_shape_{1,29};
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "generic_sensor_policy_player"};
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::unique_ptr<Ort::Session> depth_sess_, actor_sess_;
  std::string depth_input_name_, depth_output_name_, actor_input_name_, actor_output_name_;
};

class UnitreeDDS {
 public:
  UnitreeDDS(const Args& args,
             bool publish_enabled,
             std::vector<int> joint_map,
             std::vector<float> joint_signs,
             std::vector<float> lower_limits,
             std::vector<float> upper_limits)
      : args_(args),
        publish_enabled_(publish_enabled),
        joint_map_(std::move(joint_map)),
        joint_signs_(std::move(joint_signs)) {
    event_log_.open_append(args_.event_log);
    yamlc::require_permutation("robot.joint_map", joint_map_, kNumJoints);
    yamlc::require_size("robot.joint_signs", kNumJoints, joint_signs_.size());
    yamlc::require_size("robot.joint_pos_lower_limit", kNumJoints, lower_limits.size());
    yamlc::require_size("robot.joint_pos_upper_limit", kNumJoints, upper_limits.size());
    CommandSafetyConfig safety_cfg;
    safety_cfg.lowstate_timeout_sec = 0.2;
    safety_filter_.set_config(safety_cfg);
    safety_limits_.lower = std::move(lower_limits);
    safety_limits_.upper = std::move(upper_limits);
    unitree::robot::ChannelFactory::Instance()->Init(0, args_.net);
    if (publish_enabled_) { publisher_.reset(new unitree::robot::ChannelPublisher<LowCmd>(args_.cmd_topic)); publisher_->InitChannel(); }
    subscriber_.reset(new unitree::robot::ChannelSubscriber<LowState>(args_.state_topic));
    subscriber_->InitChannel(std::bind(&UnitreeDDS::on_low_state, this, std::placeholders::_1), 1);
    if (args_.use_torso_imu) {
      torso_imu_subscriber_.reset(new unitree::robot::ChannelSubscriber<IMUState>(args_.torso_imu_topic));
      torso_imu_subscriber_->InitChannel(std::bind(&UnitreeDDS::on_torso_imu, this, std::placeholders::_1), 1);
    }
  }
  void wait_for_state() {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!connected_.load() && std::chrono::steady_clock::now() < deadline && !g_stop_requested.load()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!connected_.load()) throw std::runtime_error("timeout waiting for rt/lowstate");
    if (args_.use_torso_imu && args_.require_torso_imu) {
      const auto torso_deadline = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(std::max(0.0, args_.torso_imu_timeout_sec)));
      while (!torso_imu_connected_.load() &&
             std::chrono::steady_clock::now() < torso_deadline &&
             !g_stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      if (!torso_imu_connected_.load()) {
        throw std::runtime_error("timeout waiting for torso IMU topic " + args_.torso_imu_topic);
      }
    }
    std::cout << "[generic_sensor_policy_player][instinct_parkour] lowstate connected";
    if (args_.use_torso_imu) {
      std::cout << "; torso_imu=" << (torso_imu_connected_.load() ? "connected" : "waiting/fallback")
                << " topic=" << args_.torso_imu_topic;
    } else {
      std::cout << "; torso_imu=disabled(using lowstate/pelvis IMU)";
    }
    std::cout << "\n";
  }
  RobotState snapshot() const {
    std::lock_guard<std::mutex> lk(mtx_);
    RobotState out = state_;
    if (args_.use_torso_imu && torso_imu_connected_.load()) {
      out.quat = torso_quat_;
      out.gyro = torso_gyro_;
      out.torso_imu_connected = true;
      out.imu_source = "torso";
      out.torso_imu_age_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - torso_time_).count();
    } else {
      out.imu_source = "pelvis";
    }
    return out;
  }
  void publish(const std::vector<float>& target, const std::vector<float>& kp, const std::vector<float>& kd,
               bool allow_stale_state = false, bool active_command = true) {
    if (!publish_enabled_ || !publisher_) return;
    if (!active_command && args_.experimental_no_prestart_hold) {
      if (!prestart_hold_suppressed_logged_) {
        prestart_hold_suppressed_logged_ = true;
        event_log_.log("prestart_hold_suppressed",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("variant", "instinct_parkour"),
                        event_string("reason", "experimental_handoff_ablation")});
      }
      return;
    }
    const auto s = snapshot();
    CommandSafetyState safety_state;
    safety_state.connected = s.connected;
    safety_state.stamp = s.recv_stamp;
    safety_state.seq = s.seq;
    safety_state.q = s.q_sim;
    action_go::runtime::FilteredCommand filtered;
    try {
      filtered = safety_filter_.filter(target, kp, kd, safety_state, safety_limits_, kNumJoints, allow_stale_state);
    } catch (const std::exception& e) {
      safety_failed_.store(true);
      event_log_.log("safety_stop",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_string("reason", e.what()),
                      event_int("state_seq", static_cast<long long>(s.seq))});
      std::cerr << "[generic_sensor_policy_player][instinct_parkour] safety stop: " << e.what() << "\n";
      g_stop_requested.store(true);
      return;
    }

    LowCmd cmd; cmd.mode_pr() = 0; cmd.mode_machine() = static_cast<uint8_t>(s.mode_machine > 0 ? s.mode_machine : args_.mode_machine);
    for (int sim=0; sim<kNumJoints && sim<(int)joint_map_.size(); ++sim) {
      int real = joint_map_[static_cast<size_t>(sim)]; if (real >= (int)cmd.motor_cmd().size()) continue;
      cmd.motor_cmd()[real].mode() = 1;
      cmd.motor_cmd()[real].q() = filtered.q[sim] * joint_signs_[static_cast<size_t>(sim)];
      cmd.motor_cmd()[real].dq() = 0.f; cmd.motor_cmd()[real].tau() = 0.f;
      cmd.motor_cmd()[real].kp() = filtered.kp[sim]; cmd.motor_cmd()[real].kd() = filtered.kd[sim];
    }
    cmd.crc() = crc32_core((uint32_t*)&cmd, (sizeof(cmd) >> 2) - 1);
    publisher_->Write(cmd);
    if (!first_command_logged_) {
      first_command_logged_ = true;
      event_log_.log("first_command_published",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_string("command_phase", active_command ? "active" : "hold"),
                      event_int("state_seq", static_cast<long long>(s.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
    if (!active_command && !first_hold_logged_) {
      first_hold_logged_ = true;
      event_log_.log("first_hold_published",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_int("state_seq", static_cast<long long>(s.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
    if (active_command && !first_active_command_logged_) {
      first_active_command_logged_ = true;
      event_log_.log("first_active_command",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_int("state_seq", static_cast<long long>(s.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
  }
  void publish_damping_for_stop() {
    if (!publish_enabled_ || !publisher_) return;
    auto s = snapshot();
    std::vector<float> kp(kNumJoints, 0.f), kd(kNumJoints, 3.f);
    for (int i=0;i<20;++i) {
      publish(s.q_sim, kp, kd, true, false);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  bool safety_failed() const { return safety_failed_.load(); }
 private:
  void on_torso_imu(const void* msg) {
    const auto imu = *(const IMUState*)msg;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      torso_quat_ = imu.quaternion();
      torso_gyro_ = imu.gyroscope();
      torso_time_ = std::chrono::steady_clock::now();
    }
    torso_imu_connected_.store(true);
  }
  void on_low_state(const void* msg) {
    const auto low = *(const LowState*)msg;
    if (low.motor_state().size() < kNumJoints) return;
    const auto finite = [](float value) { return std::isfinite(value); };
    const auto quat = low.imu_state().quaternion();
    const auto gyro = low.imu_state().gyroscope();
    if (!std::all_of(quat.begin(), quat.end(), finite) ||
        !std::all_of(gyro.begin(), gyro.end(), finite)) return;
    RobotState s; s.connected = true; s.recv_stamp = std::chrono::steady_clock::now(); s.seq = sequence_.fetch_add(1, std::memory_order_relaxed) + 1; s.mode_machine = (int)low.mode_machine(); s.quat = low.imu_state().quaternion(); s.gyro = low.imu_state().gyroscope();
    for (int sim=0; sim<kNumJoints; ++sim) {
      int real=joint_map_[static_cast<size_t>(sim)];
      if (real < 0 || real >= static_cast<int>(low.motor_state().size())) return;
      s.q_sim[sim]=low.motor_state()[real].q()*joint_signs_[static_cast<size_t>(sim)];
      s.dq_sim[sim]=low.motor_state()[real].dq()*joint_signs_[static_cast<size_t>(sim)];
      if (!finite(s.q_sim[sim]) || !finite(s.dq_sim[sim])) return;
    }
    const auto& r = low.wireless_remote(); for (size_t i=0; i<s.wireless_remote.size() && i<r.size(); ++i) s.wireless_remote[i] = static_cast<uint8_t>(r[i]);
    { std::lock_guard<std::mutex> lk(mtx_); state_ = s; }
    connected_.store(true);
  }
  Args args_; bool publish_enabled_{false}; mutable std::mutex mtx_; RobotState state_;
  std::vector<int> joint_map_;
  std::vector<float> joint_signs_;
  CommandSafetyFilter safety_filter_;
  CommandSafetyLimits safety_limits_;
  RuntimeEventLogger event_log_;
  bool first_command_logged_{false};
  bool first_hold_logged_{false};
  bool first_active_command_logged_{false};
  bool prestart_hold_suppressed_logged_{false};
  std::atomic<bool> safety_failed_{false};
  std::array<float, 4> torso_quat_{1.f,0.f,0.f,0.f};
  std::array<float, 3> torso_gyro_{0.f,0.f,0.f};
  std::chrono::steady_clock::time_point torso_time_{};
  std::atomic<bool> connected_{false}, torso_imu_connected_{false};
  std::atomic<uint64_t> sequence_{0};
  unitree::robot::ChannelPublisherPtr<LowCmd> publisher_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<IMUState>> torso_imu_subscriber_;
};

struct RunStats { int count{0}; bool nonfinite{false}; float max_action_abs{0}, max_target_err{0}, max_dq{0}, max_tilt{0}; };

class Runtime {
 public:
  explicit Runtime(Args args) : args_(std::move(args)), policy_(args_) {
    event_log_.open_append(args_.event_log);
    if (!args_.standdir.empty()) {
      Args stand_args = args_;
      stand_args.logdir = args_.standdir;
      stand_policy_ = std::make_unique<InstinctPolicy>(stand_args);
    }
  }
  int run() {
    if (args_.check_only) {
      policy_.check();
      check_stand_policy();
      event_log_.log("model_loaded",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_string("model", args_.logdir)});
      event_log_.log("check_only_ok",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour")});
      return 0;
    }
    if (args_.dry_run && !args_.camera_check) { policy_.check(); check_stand_policy(); depth_ = std::make_unique<ZeroDepthSource>(policy_.depth_config()); inference_loop(nullptr); std::cout << "[generic_sensor_policy_player][instinct_parkour] dry-run completed\n"; return 0; }
    if (args_.dds_dry_run) {
      policy_.check();
      check_stand_policy();
      dds_ = make_dds(false);
      dds_->wait_for_state();
      init_depth();
      inference_loop(dds_.get());
      check_preflight();
      std::cout << "[generic_sensor_policy_player][instinct_parkour] DDS dry-run completed\n";
      return 0;
    }
    if (args_.camera_check) { policy_.check(); check_stand_policy(); init_depth(); inference_loop(nullptr); std::cout << "[generic_sensor_policy_player][instinct_parkour] camera-check completed\n"; return 0; }
    real_run(); return 0;
  }
 private:
  void check_stand_policy() const {
    if (!stand_policy_) return;
    std::cout << "[generic_sensor_policy_player][instinct_parkour] stand policy configured: "
              << fs::absolute(args_.standdir) << "\n";
    stand_policy_->check();
  }
  RobotState fake_state() const { RobotState s; s.connected=true; s.mode_machine=args_.mode_machine; s.q_sim = policy_.default_joint_pos(); return s; }
  std::unique_ptr<UnitreeDDS> make_dds(bool publish_enabled) const {
    return std::make_unique<UnitreeDDS>(args_, publish_enabled,
                                        policy_.joint_map(),
                                        policy_.joint_signs(),
                                        policy_.lower_limits(),
                                        policy_.upper_limits());
  }
  void init_depth() { if (depth_) return; if (args_.depth_source == "zero") depth_ = std::make_unique<ZeroDepthSource>(policy_.depth_config()); else depth_ = std::make_unique<RealSenseDepthSource>(policy_.depth_config(), args_); }
  std::vector<float> limit_target_rate(const std::vector<float>& target, const std::vector<float>& initial) {
    if (args_.target_rate_limit <= 1e-6) return target;
    const auto now = std::chrono::steady_clock::now();
    if (filtered_target_.empty()) { filtered_target_ = initial; filtered_time_ = now; }
    const double nominal = 1.0 / std::max(1, args_.hz);
    double dt = std::chrono::duration<double>(now - filtered_time_).count(); dt = std::clamp(dt, 0.5*nominal, 0.1);
    const float max_step = static_cast<float>(args_.target_rate_limit * dt);
    std::vector<float> out(kNumJoints);
    for (int i=0;i<kNumJoints;++i) out[i] = filtered_target_[i] + std::clamp(target[i]-filtered_target_[i], -max_step, max_step);
    filtered_target_=out; filtered_time_=now; return out;
  }
  void update_stats(const RobotState& s, const std::vector<float>& action, const std::vector<float>& target) {
    ++stats_.count; stats_.max_tilt=std::max(stats_.max_tilt, tilt_deg_from_quat(s.quat));
    for (int i=0;i<kNumJoints;++i) { stats_.max_action_abs=std::max(stats_.max_action_abs, std::abs(action[i])); stats_.max_target_err=std::max(stats_.max_target_err, std::abs(target[i]-s.q_sim[i])); stats_.max_dq=std::max(stats_.max_dq, std::abs(s.dq_sim[i])); }
  }
  void inference_loop(UnitreeDDS* dds) {
    init_depth();
    const auto start = std::chrono::steady_clock::now(); auto next = start; int count=0; double avg=0, mx=0;
    while (!g_stop_requested.load()) {
      if (args_.duration_sec > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() >= args_.duration_sec) break;
      auto t0=std::chrono::steady_clock::now(); RobotState s = dds ? dds->snapshot() : fake_state();
      Command cmd = joystick_command(s.wireless_remote, args_); auto depth = depth_->observation(); auto action = policy_.infer(s, cmd, depth); auto target = policy_.action_to_target(action, s); target = limit_target_rate(target, s.q_sim);
      update_stats(s, action, target); set_latest(target, policy_.kp(), policy_.kd());
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); ++count; avg += (ms-avg)/count; mx=std::max(mx,ms);
      if (count % std::max(1,args_.stats_interval)==0) {
        auto [mn,mxv,mean] = min_max_mean(depth);
        auto [action_abs, action_idx] = max_abs(action);
        auto [target_err, target_err_idx] = max_abs_diff(target, s.q_sim);
        auto [q_default_err, q_default_idx] = max_abs_diff(s.q_sim, policy_.default_joint_pos());
        std::cout << std::fixed << std::setprecision(3) << "[generic_sensor_policy_player][instinct_parkour] infer avg=" << avg << "ms max=" << mx << "ms action=[" << *std::min_element(action.begin(),action.end()) << "," << *std::max_element(action.begin(),action.end()) << "] cmd=(" << cmd.vx << "," << cmd.vy << "," << cmd.yaw << ") tilt=" << std::setprecision(1) << tilt_deg_from_quat(s.quat) << "deg dq_max=" << stats_.max_dq << std::setprecision(2) << " depth_obs=[" << mn << "," << mxv << "] depth_mean=" << mean << " " << depth_->stats() << "\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "[generic_sensor_policy_player][instinct_parkour] diag imu=" << s.imu_source
                  << (s.torso_imu_connected ? (" torso_age=" + std::to_string(static_cast<int>(s.torso_imu_age_ms)) + "ms") : "")
                  << " action_abs=" << action_abs << "@" << policy_.joint_name(action_idx)
                  << " target_err=" << target_err << "@" << policy_.joint_name(target_err_idx)
                  << " qerr_default=" << q_default_err << "@" << policy_.joint_name(q_default_idx)
                  << " max_target_err_seen=" << stats_.max_target_err
                  << "\n";
        event_log_.log("runtime_metrics",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("template", "depth_conditioned_actor"),
                        event_string("variant", "instinct_parkour"),
                        event_int("step", count),
                        event_number("infer_ms_avg", avg),
                        event_number("infer_ms_max", mx),
                        event_number("action_abs_max", action_abs),
                        event_number("target_err_max", target_err),
                        event_number("tilt_deg", tilt_deg_from_quat(s.quat)),
                        event_number("depth_mean", mean),
                        event_int("policy_deadline_miss", policy_deadline_miss_.load()),
                        event_int("publish_deadline_miss", publish_deadline_miss_.load())});
        mx=0;
      }
      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0/std::max(1,args_.hz))); auto now=std::chrono::steady_clock::now(); if (next < now) { ++policy_deadline_miss_; next=now; } std::this_thread::sleep_until(next);
    }
  }
  static std::tuple<float,float,float> min_max_mean(const std::vector<float>& v) { if (v.empty()) return {0,0,0}; auto mm=std::minmax_element(v.begin(),v.end()); float sum=std::accumulate(v.begin(),v.end(),0.f); return {*mm.first,*mm.second,sum/v.size()}; }
  static std::vector<float> scaled_clamped(const std::vector<float>& in, float factor, float lo, float hi) {
    std::vector<float> out = in;
    for (float& v : out) v = std::clamp(v * factor, lo, hi);
    return out;
  }
  static std::pair<float,int> max_abs(const std::vector<float>& v) {
    float best = 0.f; int idx = 0;
    for (int i=0; i<(int)v.size(); ++i) {
      const float a = std::abs(v[static_cast<size_t>(i)]);
      if (a > best) { best = a; idx = i; }
    }
    return {best, idx};
  }
  static std::pair<float,int> max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float best = 0.f; int idx = 0;
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    for (int i=0; i<n; ++i) {
      const float d = std::abs(a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]);
      if (d > best) { best = d; idx = i; }
    }
    return {best, idx};
  }
  void set_latest(const std::vector<float>& q, const std::vector<float>& kp, const std::vector<float>& kd) { std::lock_guard<std::mutex> lk(target_mtx_); latest_q_=q; latest_kp_=kp; latest_kd_=kd; ++seq_; }
  bool get_latest(std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd, uint64_t& seq) { std::lock_guard<std::mutex> lk(target_mtx_); if (latest_q_.empty()) return false; q=latest_q_; kp=latest_kp_; kd=latest_kd_; seq=seq_; return true; }
  void wait_for_start_file() {
    if (args_.start_file.empty()) return;
    std::cout << "[generic_sensor_policy_player][instinct_parkour] holding current pose until start file: "
              << args_.start_file << "\n";
    while (!g_stop_requested.load() && !fs::exists(args_.start_file)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  void publish_loop() {
    std::vector<float> q,kp,kd,last_q,from,to; uint64_t seq=0,last_seq=0; get_latest(q,kp,kd,seq); from=to=q; last_q=q; auto interp_start=std::chrono::steady_clock::now(); auto next=std::chrono::steady_clock::now();
    while (!g_stop_requested.load() && !publish_stop_.load()) {
      if (get_latest(q,kp,kd,seq) && seq!=last_seq) { from=last_q.empty()?q:last_q; to=q; interp_start=std::chrono::steady_clock::now(); last_seq=seq; }
      double a = args_.target_interp_sec > 1e-6 ? std::chrono::duration<double>(std::chrono::steady_clock::now()-interp_start).count()/args_.target_interp_sec : 1.0; a=std::clamp(a,0.0,1.0); last_q=from; if (from.size()==to.size()) for(size_t i=0;i<from.size();++i) last_q[i]=static_cast<float>(from[i]+a*(to[i]-from[i])); else last_q=to;
      const bool waiting_for_start = !args_.start_file.empty() && !fs::exists(args_.start_file);
      dds_->publish(last_q,kp,kd,waiting_for_start,!waiting_for_start); next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0/std::max(args_.publish_hz,args_.hz))); auto now=std::chrono::steady_clock::now(); if(next<now) { ++publish_deadline_miss_; next=now; } std::this_thread::sleep_until(next);
    }
  }
  void run_startup_transition() {
    const bool use_step = args_.startup_step_size > 1e-6;
    const bool use_time = args_.startup_sec > 1e-6;
    const bool use_hold = args_.startup_hold_sec > 1e-6;
    if (!use_step && !use_time && !use_hold) return;

    const auto default_q = policy_.default_joint_pos();
    const auto initial_state = dds_->snapshot();
    const auto [initial_err, initial_idx] = max_abs_diff(initial_state.q_sim, default_q);
    std::cout << "[generic_sensor_policy_player][instinct_parkour] startup transition to policy default: initial_qerr="
              << std::fixed << std::setprecision(3) << initial_err << "@"
              << policy_.joint_name(initial_idx);
    if (use_step) {
      std::cout << " step_size=" << args_.startup_step_size;
    } else if (use_time) {
      std::cout << " ramp_sec=" << args_.startup_sec;
    }
    if (use_hold) std::cout << " hold_sec=" << args_.startup_hold_sec;
    std::cout << "\n";

    std::vector<float> last_target = initial_state.q_sim;
    if (use_step) {
      auto next = std::chrono::steady_clock::now();
      const double max_sec = std::max(0.5, std::min(3.0, std::ceil(initial_err / args_.startup_step_size) / std::max(1, args_.hz) + 1.0));
      const auto deadline = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(max_sec));
      while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
        const auto s = dds_->snapshot();
        last_target.resize(kNumJoints);
        float max_err = 0.f;
        for (int i = 0; i < kNumJoints; ++i) {
          const float err = default_q[static_cast<size_t>(i)] - s.q_sim[static_cast<size_t>(i)];
          max_err = std::max(max_err, std::abs(err));
          const float step = static_cast<float>(args_.startup_step_size);
          last_target[static_cast<size_t>(i)] =
              s.q_sim[static_cast<size_t>(i)] + std::clamp(err, -step, step);
        }
        set_latest(last_target, policy_.kp(), policy_.kd());
        filtered_target_ = last_target;
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / std::max(1, args_.hz)));
        std::this_thread::sleep_until(next);
        if (max_err <= static_cast<float>(args_.startup_step_size) + 1e-3f) break;
      }
      set_latest(default_q, policy_.kp(), policy_.kd());
      filtered_target_ = default_q;
      last_target = default_q;
    } else if (use_time) {
      const auto from = initial_state.q_sim;
      const auto start = std::chrono::steady_clock::now();
      const auto end = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(args_.startup_sec));
      auto next = start;
      while (!g_stop_requested.load() && std::chrono::steady_clock::now() < end) {
        const double a = std::clamp(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() /
                                    std::max(1e-6, args_.startup_sec), 0.0, 1.0);
        last_target.resize(kNumJoints);
        for (int i = 0; i < kNumJoints; ++i) {
          last_target[static_cast<size_t>(i)] = static_cast<float>(
              from[static_cast<size_t>(i)] + a * (default_q[static_cast<size_t>(i)] - from[static_cast<size_t>(i)]));
        }
        set_latest(last_target, policy_.kp(), policy_.kd());
        filtered_target_ = last_target;
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / std::max(1, args_.hz)));
        std::this_thread::sleep_until(next);
      }
      set_latest(default_q, policy_.kp(), policy_.kd());
      filtered_target_ = default_q;
      last_target = default_q;
    }

    if (use_hold && !g_stop_requested.load()) {
      set_latest(last_target, policy_.kp(), policy_.kd());
      std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(args_.startup_hold_sec)));
    }
    const auto final_state = dds_->snapshot();
    const auto [final_err, final_idx] = max_abs_diff(final_state.q_sim, default_q);
    std::cout << "[generic_sensor_policy_player][instinct_parkour] startup transition done: qerr="
              << std::fixed << std::setprecision(3) << final_err << "@"
              << policy_.joint_name(final_idx) << "\n";
  }
  void run_official_cold_start_before_stand() {
    if (!args_.cold_start_before_stand) return;
    const bool will_run_stand =
        stand_policy_ && (args_.stand_until_l1 || args_.stand_warmup_sec > 1e-6);
    const char* next_stage = will_run_stand ? "stand" : "parkour";
    const auto target_q = policy_.default_joint_pos();
    auto kp = scaled_clamped(policy_.kp(), args_.cold_start_kpkd_factor, 0.f, 500.f);
    auto kd = scaled_clamped(policy_.kd(), args_.cold_start_kpkd_factor, 0.f, 5.f);
    const float step = static_cast<float>(std::max(1e-4, args_.cold_start_step_size));

    auto s0 = dds_->snapshot();
    auto [initial_err, initial_idx] = max_abs_diff(s0.q_sim, target_q);
    std::cout << "[generic_sensor_policy_player][instinct_parkour] official cold_start before " << next_stage
              << ": initial_qerr="
              << std::fixed << std::setprecision(3) << initial_err << "@"
              << policy_.joint_name(initial_idx)
              << " step_size=" << step
              << " kpkd_factor=" << args_.cold_start_kpkd_factor << "\n";

    auto next = std::chrono::steady_clock::now();
    int count = 0;
    std::vector<float> last_action(kNumJoints, 0.f);
    while (!g_stop_requested.load()) {
      const auto s = dds_->snapshot();
      std::vector<float> desired(kNumJoints, 0.f);
      bool done = true;
      float max_err = 0.f;
      int max_idx = 0;
      for (int i = 0; i < kNumJoints; ++i) {
        const float err = target_q[static_cast<size_t>(i)] - s.q_sim[static_cast<size_t>(i)];
        const float aerr = std::abs(err);
        if (aerr > max_err) { max_err = aerr; max_idx = i; }
        if (aerr > step) {
          done = false;
          desired[static_cast<size_t>(i)] = s.q_sim[static_cast<size_t>(i)] + std::copysign(step, err);
        } else {
          desired[static_cast<size_t>(i)] = target_q[static_cast<size_t>(i)];
        }
      }

      last_action = policy_.target_to_action(desired);
      auto clipped_target = policy_.action_to_target(last_action, s, &kp, &kd);
      set_latest(clipped_target, kp, kd);
      filtered_target_ = clipped_target;
      policy_.seed_last_action(last_action);
      if (stand_policy_) stand_policy_->seed_last_action(last_action);

      ++count;
      if (count % std::max(1, args_.stats_interval) == 0 || done) {
        std::cout << "[generic_sensor_policy_player][instinct_parkour] cold_start diag qerr="
                  << std::fixed << std::setprecision(3) << max_err << "@"
                  << policy_.joint_name(max_idx) << "\n";
      }
      if (done) break;

      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / std::max(1, args_.hz)));
      const auto now = std::chrono::steady_clock::now();
      if (next < now) next = now;
      std::this_thread::sleep_until(next);
    }

    if (!g_stop_requested.load() && args_.cold_start_hold_sec > 1e-6) {
      const auto zero_action = policy_.target_to_action(target_q);
      set_latest(target_q, kp, kd);
      filtered_target_ = target_q;
      policy_.seed_last_action(zero_action);
      if (stand_policy_) stand_policy_->seed_last_action(zero_action);
      std::cout << "[generic_sensor_policy_player][instinct_parkour] official cold_start hold default for "
                << std::fixed << std::setprecision(3) << args_.cold_start_hold_sec
                << "s before " << next_stage << "\n";
      std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(args_.cold_start_hold_sec)));
    }
    const auto sf = dds_->snapshot();
    auto [final_err, final_idx] = max_abs_diff(sf.q_sim, target_q);
    auto [final_dq, final_dq_idx] = max_abs(sf.dq_sim);
    std::cout << "[generic_sensor_policy_player][instinct_parkour] official cold_start done: qerr="
              << std::fixed << std::setprecision(3) << final_err << "@"
              << policy_.joint_name(final_idx)
              << " dq=" << final_dq << "@"
              << policy_.joint_name(final_dq_idx) << "\n";
  }
  void run_stand_policy_stage() {
    if (!stand_policy_) return;
    if (!args_.stand_until_l1 && args_.stand_warmup_sec <= 1e-6) return;

    std::cout << "[generic_sensor_policy_player][instinct_parkour] running official stand policy";
    if (args_.stand_until_l1) {
      std::cout << "; press L1 to switch to parkour";
      if (args_.stand_warmup_sec > 1e-6) {
        std::cout << " (or wait " << args_.stand_warmup_sec << "s)";
      }
    } else {
      std::cout << " for " << args_.stand_warmup_sec << "s";
    }
    std::cout << "\n";

    std::vector<float> stand_depth(static_cast<size_t>(
        stand_policy_->depth_config().num_frames *
        stand_policy_->depth_config().depth_h *
        stand_policy_->depth_config().depth_w), 0.f);
    std::vector<float> last_action(kNumJoints, 0.f);
    const auto start = std::chrono::steady_clock::now();
    auto next = start;
    int count = 0;
    while (!g_stop_requested.load()) {
      const auto s = dds_->snapshot();
      const bool l1 = remote_button(s.wireless_remote, kRemoteL1);
      const bool timeout = args_.stand_warmup_sec > 1e-6 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= args_.stand_warmup_sec;
      if (args_.stand_until_l1 && l1) {
        std::cout << "[generic_sensor_policy_player][instinct_parkour] L1 pressed, switching stand -> parkour\n";
        break;
      }
      if (!args_.stand_until_l1 && timeout) {
        break;
      }
      if (args_.stand_until_l1 && timeout) {
        std::cout << "[generic_sensor_policy_player][instinct_parkour] stand warmup timeout, switching stand -> parkour\n";
        break;
      }

      Command zero_cmd;
      auto action = stand_policy_->infer(s, zero_cmd, stand_depth);
      auto target = stand_policy_->action_to_target(action, s);
      target = limit_target_rate(target, s.q_sim);
      set_latest(target, stand_policy_->kp(), stand_policy_->kd());
      last_action = action;

      ++count;
      if (count % std::max(1, args_.stats_interval) == 0) {
        auto [action_abs, action_idx] = max_abs(action);
        auto [target_err, target_err_idx] = max_abs_diff(target, s.q_sim);
        auto [q_default_err, q_default_idx] = max_abs_diff(s.q_sim, stand_policy_->default_joint_pos());
        std::cout << std::fixed << std::setprecision(3)
                  << "[generic_sensor_policy_player][instinct_parkour] stand diag imu=" << s.imu_source
                  << " action_abs=" << action_abs << "@" << policy_.joint_name(action_idx)
                  << " target_err=" << target_err << "@" << policy_.joint_name(target_err_idx)
                  << " qerr_default=" << q_default_err << "@" << policy_.joint_name(q_default_idx)
                  << "\n";
      }

      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / std::max(1, args_.hz)));
      const auto now = std::chrono::steady_clock::now();
      if (next < now) next = now;
      std::this_thread::sleep_until(next);
    }
    policy_.seed_last_action(last_action);
  }
  void log_parkour_depth_handoff() {
    if (!depth_) return;
    try {
      const auto depth = depth_->observation();
      const auto [mn, mxv, mean] = min_max_mean(depth);
      std::cout << std::fixed << std::setprecision(2)
                << "[generic_sensor_policy_player][instinct_parkour] parkour depth handoff depth_obs=["
                << mn << "," << mxv << "] depth_mean=" << mean << " "
                << depth_->stats() << "\n";
      if (depth_->frame_count() <= 0) {
        std::cout << "[generic_sensor_policy_player][instinct_parkour] WARNING: parkour starts with no RealSense frames; "
                  << "depth input will be zero until frames arrive\n";
      }
    } catch (const std::exception& e) {
      std::cout << "[generic_sensor_policy_player][instinct_parkour] WARNING: failed to read depth before parkour start: "
                << e.what() << "\n";
    }
  }
  void real_run() {
    const bool preview_mode = !args_.preview_file.empty();
    const bool prestart_handoff = !args_.start_file.empty() || preview_mode;
    dds_ = make_dds(true); dds_->wait_for_state(); auto s=dds_->snapshot(); set_latest(s.q_sim, std::vector<float>(kNumJoints,20.f), std::vector<float>(kNumJoints,1.f)); filtered_target_=s.q_sim;
    if (prestart_handoff) {
      std::cout << "[generic_sensor_policy_player][instinct_parkour] prestart: initializing depth source before UserCtrl\n";
    }
    // In the prestart path, official walkrun is still active while RealSense
    // and ONNX/DDS are initialized.  If the depth camera is missing or the
    // policy config is wrong, fail here before UserCtrl is entered.
    init_depth();
    std::optional<InstinctPolicy::StateSnapshot> preview_policy_state;
    std::vector<float> preview_target;
    if (preview_mode) {
      preview_policy_state = policy_.snapshot_state();
      const auto depth_snapshot = depth_->observation();
      const Command command_snapshot = joystick_command(s.wireless_remote, args_);
      const auto action = policy_.infer(s, command_snapshot, depth_snapshot);
      preview_target = policy_.action_to_target(action, s);
      set_latest(preview_target, policy_.kp(), policy_.kd());
      filtered_target_ = preview_target;
      event_log_.log("preview_input_snapshot",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("template", "depth_conditioned_actor"),
                      event_string("variant", "instinct_parkour"),
                      event_string("source", "depth_frame_history"),
                      event_int("input_sequence", static_cast<long long>(depth_->sequence())),
                      event_number("input_freshness_sec", depth_->age_sec()),
                      event_string("input_hash", hash_float_vector(depth_snapshot))});
    }
    publish_stop_.store(false);
    std::thread pub(&Runtime::publish_loop,this);
    // Publish at least one hold packet before telling action_go it is safe to
    // switch to UserCtrl.
    dds_->publish(s.q_sim, policy_.kp(), policy_.kd(), true, false);
    if (!args_.ready_file.empty()) {
      touch_file(args_.ready_file);
      event_log_.log("ready_file_touched",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_string("path", args_.ready_file)});
      std::cout << "[generic_sensor_policy_player][instinct_parkour] ready file touched: "
                << args_.ready_file << "\n";
    }
    if (preview_mode) {
      const auto decision = action_go::runtime::wait_preview_lifecycle(
          args_.preview_file, args_.commit_file, args_.discard_file,
          args_.preview_timeout_sec, std::max(args_.publish_hz, args_.hz),
          []() { return g_stop_requested.load(); }, []() {});
      if (decision != action_go::runtime::PreviewLifecycleDecision::kCommit) {
        publish_stop_.store(true);
        if (pub.joinable()) pub.join();
        if (preview_policy_state) policy_.restore_state(*preview_policy_state);
        event_log_.log("preview_state_restored",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("template", "depth_conditioned_actor"),
                        event_string("variant", "instinct_parkour")});
        if (decision == action_go::runtime::PreviewLifecycleDecision::kTimeout) {
          throw std::runtime_error("transactional preview timed out before commit");
        }
        return;
      }
      event_log_.log("preview_commit_first_command_consistent",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_string("candidate_hash", hash_float_vector(preview_target))});
      std::this_thread::sleep_for(std::chrono::duration<double>(
          1.0 / std::max(1, args_.hz)));
    } else {
      wait_for_start_file();
    }
    if (!args_.start_file.empty() && fs::exists(args_.start_file)) {
      event_log_.log("start_file_seen",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "instinct_parkour"),
                      event_string("path", args_.start_file)});
    }
    if (g_stop_requested.load()) {
      if(pub.joinable()) pub.join();
      if(!args_.no_damping_on_stop) dds_->publish_damping_for_stop();
      if (dds_->safety_failed()) throw std::runtime_error("CommandSafetyFilter stopped Instinct publishing");
      return;
    }
    if (!preview_mode) {
      run_official_cold_start_before_stand();
      run_startup_transition();
      run_stand_policy_stage();
    }
    log_parkour_depth_handoff();

    std::string inference_error;
    std::atomic<bool> inference_failed{false};
    std::thread inf([&](){
      try {
        inference_loop(dds_.get());
      } catch (const std::exception& e) {
        inference_error = e.what();
        inference_failed.store(true);
        g_stop_requested.store(true);
      }
    });
    std::cout << "[generic_sensor_policy_player][instinct_parkour] start real DDS depth_source=" << args_.depth_source << " hz=" << args_.hz << " command_source=" << args_.command_source << " target_rate_limit=" << args_.target_rate_limit << " prestart=" << (prestart_handoff?"true":"false") << "\n";
    const auto start=std::chrono::steady_clock::now(); while(!g_stop_requested.load()) { if(args_.duration_sec>0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>=args_.duration_sec) break; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
    g_stop_requested.store(true); publish_stop_.store(true); if(inf.joinable()) inf.join(); if(pub.joinable()) pub.join(); if(!args_.no_damping_on_stop) dds_->publish_damping_for_stop(); if(dds_->safety_failed()) throw std::runtime_error("CommandSafetyFilter stopped Instinct publishing"); if(inference_failed.load()) throw std::runtime_error("inference loop failed: " + inference_error); event_log_.log("runtime_metrics", {event_string("player", "generic_sensor_policy_player"), event_string("template", "depth_conditioned_actor"), event_string("variant", "instinct_parkour"), event_int("policy_deadline_miss", policy_deadline_miss_.load()), event_int("publish_deadline_miss", publish_deadline_miss_.load())}); std::cout << "[generic_sensor_policy_player][instinct_parkour] completed\n";
  }
  void check_preflight() { if (!args_.fail_on_bad_input) return; std::vector<std::string> fail; if(stats_.count<=0) fail.push_back("no samples"); if(stats_.max_tilt>args_.max_preflight_tilt_deg) fail.push_back("tilt too large"); if(stats_.max_dq>args_.max_preflight_dq) fail.push_back("dq too large"); if(stats_.max_action_abs>args_.max_preflight_action_abs) fail.push_back("action too large"); if(stats_.max_target_err>args_.max_preflight_target_err) fail.push_back("target error too large"); if(args_.depth_source=="realsense" && depth_->frame_count()<args_.min_preflight_rs_frames) fail.push_back("not enough RealSense frames"); if(!fail.empty()) throw std::runtime_error("preflight sanity failed: " + fail.front()); }

  Args args_; InstinctPolicy policy_; std::unique_ptr<InstinctPolicy> stand_policy_; std::unique_ptr<DepthSource> depth_; std::unique_ptr<UnitreeDDS> dds_; RunStats stats_;
  RuntimeEventLogger event_log_;
  std::atomic<long long> policy_deadline_miss_{0};
  std::atomic<long long> publish_deadline_miss_{0};
  std::atomic<bool> publish_stop_{false};
  std::mutex target_mtx_; std::vector<float> latest_q_, latest_kp_, latest_kd_, filtered_target_; uint64_t seq_{0}; std::chrono::steady_clock::time_point filtered_time_{};
};

}  // namespace

int run_instinct_parkour_runtime(int argc, char** argv) {
  std::signal(SIGINT, signal_handler); std::signal(SIGTERM, signal_handler);
  try {
    Args args = parse_args(argc, argv);
    if (args.depth_camera_check_only) return run_depth_camera_check_only(args);
    Runtime rt(std::move(args));
    return rt.run();
  }
  catch (const std::exception& e) { std::cerr << "[generic_sensor_policy_player][instinct_parkour] ERROR: " << e.what() << "\n"; return 1; }
}
