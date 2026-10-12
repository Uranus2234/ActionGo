#include "action_go/runtime/activation_blend.h"
#include "action_go/runtime/command_safety_filter.h"
#include "action_go/runtime/event_logger.h"
#include "action_go/runtime/policy_protocol.h"
#include "action_go/runtime/preview_lifecycle.h"
#include "action_go/runtime/yaml_contract.h"

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#ifdef ACTION_GO_WITH_REALSENSE
#include <librealsense2/rs.hpp>
#endif

#if defined(ACTION_GO_WITH_APRILTAG)
#if __has_include(<apriltag.h>)
#include <apriltag.h>
#include <apriltag_pose.h>
#include <tagStandard41h12.h>
#elif __has_include(<apriltag/apriltag.h>)
#include <apriltag/apriltag.h>
#include <apriltag/apriltag_pose.h>
#include <apriltag/tagStandard41h12.h>
#else
#undef ACTION_GO_WITH_APRILTAG
#endif
#endif

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

namespace {
std::atomic<bool> g_stop_requested{false};
void signal_handler(int) { g_stop_requested.store(true); }

using action_go::runtime::CommandSafetyConfig;
using action_go::runtime::ActivationBlend;
using action_go::runtime::JointCommand;
using action_go::runtime::CommandSafetyFilter;
using action_go::runtime::CommandSafetyLimits;
using action_go::runtime::CommandSafetyState;
using action_go::runtime::RuntimeEventLogger;
using action_go::runtime::event_int;
using action_go::runtime::event_number;
using action_go::runtime::event_string;
using action_go::runtime::kSteadyTrayProtocolEncoderHistoryLen;
using action_go::runtime::kSteadyTrayProtocolFrameDim;
using action_go::runtime::kSteadyTrayProtocolPolicyFrames;
using action_go::runtime::kSteadyTrayProtocolObsDim;
using action_go::runtime::kUnitreeG1ProtocolMotorCount;
namespace yamlc = action_go::runtime::yaml_contract;

constexpr int kNumJoints = kUnitreeG1ProtocolMotorCount;
constexpr int kPropObsDim = kSteadyTrayProtocolFrameDim;
constexpr int kPolicyFrames = kSteadyTrayProtocolPolicyFrames;
constexpr int kPolicyObsDim = kSteadyTrayProtocolObsDim;
constexpr int kObjectObsDim = 7;
constexpr int kStudentObsDim = kPropObsDim + kObjectObsDim;
constexpr int kEncoderSeqLen = kSteadyTrayProtocolEncoderHistoryLen;
constexpr int kDefaultModeMachine = 5;

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  constexpr uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1u << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
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

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

template <typename T>
std::vector<T> read_vec(const YAML::Node& root, const std::string& key) {
  if (!root[key]) throw std::runtime_error("missing params key: " + key);
  std::vector<T> out;
  for (const auto& v : root[key]) out.push_back(v.as<T>());
  return out;
}

template <typename T>
std::vector<T> read_vec_any(const YAML::Node& root, std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (root[key]) return read_vec<T>(root, key);
  }
  std::ostringstream ss;
  ss << "missing params key; tried";
  for (const char* key : keys) ss << " " << key;
  throw std::runtime_error(ss.str());
}

void check_size(const std::string& name, size_t expect, size_t actual) {
  if (expect != actual) {
    throw std::runtime_error(name + " size mismatch expect=" + std::to_string(expect) +
                             " actual=" + std::to_string(actual));
  }
}

std::array<float, 4> normalize_quat(std::array<float, 4> q) {
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(n > 1e-6f) || !std::isfinite(n)) return {1.f, 0.f, 0.f, 0.f};
  for (float& v : q) v /= n;
  return q;
}

std::array<float, 4> quat_mul(const std::array<float, 4>& a_in, const std::array<float, 4>& b_in) {
  const auto a = normalize_quat(a_in);
  const auto b = normalize_quat(b_in);
  const float w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3];
  const float w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];
  return normalize_quat({
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  });
}

std::array<float, 9> matrix_from_quat(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  return {
      1.f - 2.f * (y * y + z * z), 2.f * (x * y - z * w),       2.f * (x * z + y * w),
      2.f * (x * y + z * w),       1.f - 2.f * (x * x + z * z), 2.f * (y * z - x * w),
      2.f * (x * z - y * w),       2.f * (y * z + x * w),       1.f - 2.f * (x * x + y * y)};
}

std::array<float, 3> mat_apply(const std::array<float, 9>& R, const std::array<float, 3>& v) {
  return {R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
          R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
          R[6] * v[0] + R[7] * v[1] + R[8] * v[2]};
}

std::array<float, 4> quat_from_matrix(const std::array<double, 9>& R) {
  const double tr = R[0] + R[4] + R[8];
  double qw, qx, qy, qz;
  if (tr > 0.0) {
    const double s = std::sqrt(tr + 1.0) * 2.0;
    qw = 0.25 * s;
    qx = (R[7] - R[5]) / s;
    qy = (R[2] - R[6]) / s;
    qz = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
    qw = (R[7] - R[5]) / s;
    qx = 0.25 * s;
    qy = (R[1] + R[3]) / s;
    qz = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
    qw = (R[2] - R[6]) / s;
    qx = (R[1] + R[3]) / s;
    qy = 0.25 * s;
    qz = (R[5] + R[7]) / s;
  } else {
    const double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
    qw = (R[3] - R[1]) / s;
    qx = (R[2] + R[6]) / s;
    qy = (R[5] + R[7]) / s;
    qz = 0.25 * s;
  }
  return normalize_quat({static_cast<float>(qw), static_cast<float>(qx),
                         static_cast<float>(qy), static_cast<float>(qz)});
}

std::array<float, 3> projected_gravity_from_quat_steadytray(const std::array<float, 4>& quat_wxyz) {
  // Exact formula used in SteadyTray deploy/scripts/policy_runner.py:get_gravity_orientation.
  const float qw = quat_wxyz[0], qx = quat_wxyz[1], qy = quat_wxyz[2], qz = quat_wxyz[3];
  return {2.f * (-qz * qx + qw * qy),
          -2.f * (qz * qy + qw * qx),
          1.f - 2.f * (qw * qw + qz * qz)};
}

struct Args {
  std::string net{"enP8p1s0"};
  std::string model;
  std::string params;
  std::string cmd_topic{"rt/user_lowcmd"};
  std::string state_topic{"rt/lowstate"};
  std::string ready_file;
  std::string start_file;
  std::string preview_file;
  std::string commit_file;
  std::string discard_file;
  std::string event_log;
  std::string object_source{"apriltag"};  // apriltag / udp / zero
  int udp_port{39001};
  int tag_id{-1};
  float tag_size{0.08f};
  int rs_width{640};
  int rs_height{480};
  int rs_fps{30};
  int hz{50};
  int publish_hz{500};
  int onnx_threads{2};
  int mode_machine{kDefaultModeMachine};
  int stats_interval{50};
  int warmup_steps{5};
  double duration_sec{0.0};
  double target_interp_sec{0.02};
  double activation_blend_sec{0.10};
  double preview_timeout_sec{5.0};
  double pose_timeout_sec{0.15};
  double camera_timeout_sec{2.0};
  std::array<float, 3> cmd{0.5f, 0.f, 0.f};
  std::array<float, 3> vision_to_policy_pos{0.f, 0.f, 0.f};
  std::array<float, 4> vision_to_policy_quat{1.f, 0.f, 0.f, 0.f};
  std::array<float, 3> tag_to_object_pos{0.f, 0.f, 0.f};
  std::array<float, 4> tag_to_object_quat{1.f, 0.f, 0.f, 0.f};
  bool require_tag{false};
  bool exit_on_tag_lost{true};
  bool check_only{false};
  bool camera_check{false};
  bool dry_run{false};
  bool no_damping_on_stop{true};
  bool experimental_no_prestart_hold{false};
  bool cmd_overridden{false};
};

std::string need_arg(int& i, int argc, char** argv, const std::string& key) {
  if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
  return argv[++i];
}

std::array<float, 3> parse_vec3(const std::string& s) {
  std::array<float, 3> out{0.f, 0.f, 0.f};
  std::string t = s;
  for (char& c : t) if (c == ',' || c == '[' || c == ']') c = ' ';
  std::stringstream ss(t);
  for (float& v : out) ss >> v;
  return out;
}

std::array<float, 4> parse_quat(const std::string& s) {
  std::array<float, 4> out{1.f, 0.f, 0.f, 0.f};
  std::string t = s;
  for (char& c : t) if (c == ',' || c == '[' || c == ']') c = ' ';
  std::stringstream ss(t);
  for (float& v : out) ss >> v;
  return normalize_quat(out);
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--net") a.net = need_arg(i, argc, argv, k);
    else if (k == "--model") a.model = need_arg(i, argc, argv, k);
    else if (k == "--params") a.params = need_arg(i, argc, argv, k);
    else if (k == "--cmd-topic") a.cmd_topic = need_arg(i, argc, argv, k);
    else if (k == "--state-topic") a.state_topic = need_arg(i, argc, argv, k);
    else if (k == "--ready-file") a.ready_file = need_arg(i, argc, argv, k);
    else if (k == "--start-file") a.start_file = need_arg(i, argc, argv, k);
    else if (k == "--preview-file") a.preview_file = need_arg(i, argc, argv, k);
    else if (k == "--commit-file") a.commit_file = need_arg(i, argc, argv, k);
    else if (k == "--discard-file") a.discard_file = need_arg(i, argc, argv, k);
    else if (k == "--event-log") a.event_log = need_arg(i, argc, argv, k);
    else if (k == "--object-source") a.object_source = need_arg(i, argc, argv, k);
    else if (k == "--udp-port") a.udp_port = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--tag-id") a.tag_id = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--tag-size") a.tag_size = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--rs-width") a.rs_width = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--rs-height") a.rs_height = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--rs-fps") a.rs_fps = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--hz") a.hz = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--publish-hz") a.publish_hz = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--onnx-threads") a.onnx_threads = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--mode-machine") a.mode_machine = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--stats-interval") a.stats_interval = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--warmup-steps") a.warmup_steps = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--duration-sec") a.duration_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--target-interp-sec") a.target_interp_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--activation-blend-sec") a.activation_blend_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--preview-timeout-sec") a.preview_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--pose-timeout-sec") a.pose_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--camera-timeout-sec") a.camera_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--cmd" || k == "--cmd-init") { a.cmd = parse_vec3(need_arg(i, argc, argv, k)); a.cmd_overridden = true; }
    else if (k == "--cmd-vx") { a.cmd[0] = std::stof(need_arg(i, argc, argv, k)); a.cmd_overridden = true; }
    else if (k == "--cmd-vy") { a.cmd[1] = std::stof(need_arg(i, argc, argv, k)); a.cmd_overridden = true; }
    else if (k == "--cmd-yaw") { a.cmd[2] = std::stof(need_arg(i, argc, argv, k)); a.cmd_overridden = true; }
    else if (k == "--vision-to-policy-pos") a.vision_to_policy_pos = parse_vec3(need_arg(i, argc, argv, k));
    else if (k == "--vision-to-policy-quat") a.vision_to_policy_quat = parse_quat(need_arg(i, argc, argv, k));
    else if (k == "--tag-to-object-pos") a.tag_to_object_pos = parse_vec3(need_arg(i, argc, argv, k));
    else if (k == "--tag-to-object-quat") a.tag_to_object_quat = parse_quat(need_arg(i, argc, argv, k));
    else if (k == "--require-tag") a.require_tag = true;
    else if (k == "--allow-tag-lost") a.exit_on_tag_lost = false;
    else if (k == "--check-only") a.check_only = true;
    else if (k == "--camera-check") { a.camera_check = true; if (a.duration_sec <= 0.0) a.duration_sec = 5.0; }
    else if (k == "--dry-run" || k == "--dds-dry-run") a.dry_run = true;
    else if (k == "--no-damping-on-stop") a.no_damping_on_stop = true;
    else if (k == "--experimental-no-prestart-hold") a.experimental_no_prestart_hold = true;
    else if (k == "--damping-on-stop") a.no_damping_on_stop = false;
    else if (k == "--help") {
      std::cout << "Usage: generic_sensor_policy_player --spec actions/steadytray/config/generic_steadytray.yaml [runtime overrides]\n"
                << "  --object-source apriltag|udp|zero\n"
                << "  --tag-size METERS --tag-id ID --require-tag\n"
                << "  --vision-to-policy-pos 'x y z' --vision-to-policy-quat 'qw qx qy qz'\n"
                << "  --camera-check                 RealSense+AprilTag/UDP pose check only\n"
                << "  --dds-dry-run | --dry-run       subscribe lowstate and run policy but do not publish LowCmd\n"
                << "  --check-only                    load model/params and exit\n"
                << "  --udp-port 39001                receive text or binary 7-float pose x y z qw qx qy qz\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }
  if (!a.camera_check && (a.model.empty() || a.params.empty())) {
    throw std::runtime_error("--model and --params are required (except --camera-check)");
  }
  if (!a.preview_file.empty() || !a.commit_file.empty() || !a.discard_file.empty()) {
    action_go::runtime::validate_preview_lifecycle(
        a.preview_file, a.commit_file, a.discard_file,
        a.preview_timeout_sec, a.cmd_topic);
  }
  return a;
}

struct ObjectPose {
  std::array<float, kObjectObsDim> obs{0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
  bool valid{false};
  double age_sec{std::numeric_limits<double>::infinity()};
  int detections{0};
  int tag_id{-1};
  float margin{0.f};
  uint64_t sequence{0};
};

class ObjectSource {
 public:
  virtual ~ObjectSource() = default;
  virtual ObjectPose latest() = 0;
  virtual std::string stats() const = 0;
};

class ZeroObjectSource final : public ObjectSource {
 public:
  ObjectPose latest() override {
    ObjectPose p;
    p.valid = true;
    p.age_sec = 0.0;
    p.sequence = 1;
    return p;
  }
  std::string stats() const override { return "object_source=zero"; }
};

class UdpObjectSource final : public ObjectSource {
 public:
  explicit UdpObjectSource(int port) : port_(port) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error("socket(AF_INET,SOCK_DGRAM) failed");
    int yes = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      const int e = errno;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("UDP bind failed port=" + std::to_string(port_) + " errno=" + std::to_string(e));
    }
    const int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
    std::cout << "[generic_sensor_policy_player][steadytray] UDP object pose listening on port " << port_ << "\n";
  }
  ~UdpObjectSource() override { if (fd_ >= 0) ::close(fd_); }

  ObjectPose latest() override {
    drain();
    std::lock_guard<std::mutex> lk(mtx_);
    ObjectPose p = pose_;
    if (have_) {
      p.valid = true;
      p.age_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_time_).count();
      p.sequence = packets_;
    }
    return p;
  }
  std::string stats() const override {
    std::lock_guard<std::mutex> lk(mtx_);
    const double age = have_ ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last_time_).count()
                             : std::numeric_limits<double>::infinity();
    std::ostringstream ss;
    ss << "object_source=udp port=" << port_ << " packets=" << packets_ << " age=" << std::fixed << std::setprecision(0) << age << "ms";
    return ss.str();
  }

 private:
  static bool parse_packet(const char* data, ssize_t n, std::array<float, 7>& out) {
    if (n == static_cast<ssize_t>(7 * sizeof(float))) {
      std::memcpy(out.data(), data, 7 * sizeof(float));
      auto q = normalize_quat({out[3], out[4], out[5], out[6]});
      out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
      return true;
    }
    std::string s(data, data + n);
    for (char& c : s) if (c == ',' || c == '[' || c == ']' || c == ';') c = ' ';
    std::stringstream ss(s);
    for (float& v : out) {
      if (!(ss >> v)) return false;
    }
    auto q = normalize_quat({out[3], out[4], out[5], out[6]});
    out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
    return true;
  }

  void drain() {
    if (fd_ < 0) return;
    char buf[512];
    while (true) {
      const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
      if (n < 0) {
        if (errno == EWOULDBLOCK || errno == EAGAIN) return;
        return;
      }
      std::array<float, 7> obs{};
      if (parse_packet(buf, n, obs)) {
        std::lock_guard<std::mutex> lk(mtx_);
        pose_.obs = obs;
        pose_.valid = true;
        pose_.detections = 1;
        have_ = true;
        ++packets_;
        last_time_ = std::chrono::steady_clock::now();
      }
    }
  }

  int port_{39001};
  int fd_{-1};
  mutable std::mutex mtx_;
  ObjectPose pose_;
  bool have_{false};
  uint64_t packets_{0};
  std::chrono::steady_clock::time_point last_time_{};
};

#if defined(ACTION_GO_WITH_REALSENSE) && defined(ACTION_GO_WITH_APRILTAG)
class AprilTagObjectSource final : public ObjectSource {
 public:
  explicit AprilTagObjectSource(const Args& args) : args_(args) {
    if (!(args_.tag_size > 0.f)) throw std::runtime_error("--tag-size must be > 0");
    tf_ = tagStandard41h12_create();
    if (!tf_) throw std::runtime_error("tagStandard41h12_create failed");
    td_ = apriltag_detector_create();
    if (!td_) throw std::runtime_error("apriltag_detector_create failed");
    apriltag_detector_add_family(td_, tf_);
    td_->quad_decimate = 1.0;
    td_->quad_sigma = 0.0;
    td_->nthreads = 2;
    td_->debug = 0;
    td_->refine_edges = 1;

    rs2::config c;
    c.enable_stream(RS2_STREAM_COLOR, args_.rs_width, args_.rs_height, RS2_FORMAT_BGR8, args_.rs_fps);
    profile_ = pipe_.start(c);
    auto sp = profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intr = sp.get_intrinsics();
    fx_ = intr.fx; fy_ = intr.fy; cx_ = intr.ppx; cy_ = intr.ppy;
    (void)pipe_.wait_for_frames(1000);
    stop_.store(false);
    thread_ = std::thread(&AprilTagObjectSource::thread_main, this);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(args_.camera_timeout_sec));
    while (std::chrono::steady_clock::now() < deadline && !g_stop_requested.load()) {
      if (latest().valid) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "[generic_sensor_policy_player][steadytray] RealSense color started resolution=(" << args_.rs_width << "," << args_.rs_height
              << ") fps=" << args_.rs_fps << " intrinsics fx=" << fx_ << " fy=" << fy_ << " cx=" << cx_ << " cy=" << cy_ << "\n";
  }

  ~AprilTagObjectSource() override {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    try { pipe_.stop(); } catch (...) {}
    if (td_) apriltag_detector_destroy(td_);
    if (tf_) tagStandard41h12_destroy(tf_);
  }

  ObjectPose latest() override {
    std::lock_guard<std::mutex> lk(mtx_);
    ObjectPose p = pose_;
    if (p.valid) p.age_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_time_).count();
    p.sequence = frames_;
    return p;
  }

  std::string stats() const override {
    std::lock_guard<std::mutex> lk(mtx_);
    const double age = pose_.valid ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last_time_).count()
                                   : std::numeric_limits<double>::infinity();
    std::ostringstream ss;
    ss << "object_source=apriltag frames=" << frames_ << " det=" << pose_.detections
       << " tag=" << pose_.tag_id << " margin=" << std::fixed << std::setprecision(1) << pose_.margin
       << " age=" << std::setprecision(0) << age << "ms";
    return ss.str();
  }

 private:
  ObjectPose pose_from_detection(apriltag_detection_t* det, int detections) {
    apriltag_detection_info_t info;
    info.det = det;
    info.tagsize = args_.tag_size;
    info.fx = fx_;
    info.fy = fy_;
    info.cx = cx_;
    info.cy = cy_;
    apriltag_pose_t pose;
    const double err = estimate_tag_pose(&info, &pose);
    (void)err;

    std::array<double, 9> Rct{};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        Rct[static_cast<size_t>(r * 3 + c)] = MATD_EL(pose.R, r, c);
      }
    }
    const std::array<float, 3> tct{static_cast<float>(MATD_EL(pose.t, 0, 0)),
                                   static_cast<float>(MATD_EL(pose.t, 1, 0)),
                                   static_cast<float>(MATD_EL(pose.t, 2, 0))};
    matd_destroy(pose.R);
    matd_destroy(pose.t);

    const auto qvt = quat_from_matrix(Rct);
    const auto Rvt = matrix_from_quat(qvt);
    const auto tag_offset_v = mat_apply(Rvt, args_.tag_to_object_pos);
    const std::array<float, 3> tvo{tct[0] + tag_offset_v[0], tct[1] + tag_offset_v[1], tct[2] + tag_offset_v[2]};
    const auto qvo = quat_mul(qvt, args_.tag_to_object_quat);

    const auto Rpv = matrix_from_quat(args_.vision_to_policy_quat);
    const auto offset_p = mat_apply(Rpv, tvo);
    const auto qpo = quat_mul(args_.vision_to_policy_quat, qvo);

    ObjectPose out;
    out.valid = true;
    out.obs = {args_.vision_to_policy_pos[0] + offset_p[0],
               args_.vision_to_policy_pos[1] + offset_p[1],
               args_.vision_to_policy_pos[2] + offset_p[2],
               qpo[0], qpo[1], qpo[2], qpo[3]};
    out.detections = detections;
    out.tag_id = det->id;
    out.margin = det->decision_margin;
    out.age_sec = 0.0;
    return out;
  }

  void thread_main() {
    while (!stop_.load() && !g_stop_requested.load()) {
      try {
        auto frameset = pipe_.wait_for_frames(std::max(1, int(2000 / std::max(1, args_.rs_fps))));
        auto cf = frameset.get_color_frame();
        if (!cf) continue;
        cv::Mat bgr(cv::Size(args_.rs_width, args_.rs_height), CV_8UC3, (void*)cf.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        image_u8_t im;
        im.width = gray.cols;
        im.height = gray.rows;
        im.stride = static_cast<int>(gray.step[0]);
        im.buf = gray.data;
        zarray_t* detections = apriltag_detector_detect(td_, &im);
        const int ndet = zarray_size(detections);
        apriltag_detection_t* best = nullptr;
        for (int i = 0; i < ndet; ++i) {
          apriltag_detection_t* det = nullptr;
          zarray_get(detections, i, &det);
          if (!det) continue;
          if (args_.tag_id >= 0 && det->id != args_.tag_id) continue;
          if (!best || det->decision_margin > best->decision_margin) best = det;
        }
        {
          std::lock_guard<std::mutex> lk(mtx_);
          ++frames_;
          if (best) {
            pose_ = pose_from_detection(best, ndet);
            last_time_ = std::chrono::steady_clock::now();
          } else {
            pose_.detections = ndet;
          }
        }
        apriltag_detections_destroy(detections);
      } catch (const std::exception& e) {
        std::cerr << "[generic_sensor_policy_player][steadytray] RealSense/AprilTag warning: " << e.what() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  }

  Args args_;
  float fx_{0.f}, fy_{0.f}, cx_{0.f}, cy_{0.f};
  rs2::pipeline pipe_;
  rs2::pipeline_profile profile_;
  apriltag_family_t* tf_{nullptr};
  apriltag_detector_t* td_{nullptr};
  std::thread thread_;
  std::atomic<bool> stop_{false};
  mutable std::mutex mtx_;
  ObjectPose pose_;
  std::chrono::steady_clock::time_point last_time_{};
  uint64_t frames_{0};
};
#else
class AprilTagObjectSource final : public ObjectSource {
 public:
  explicit AprilTagObjectSource(const Args&) {
#if !defined(ACTION_GO_WITH_REALSENSE)
    throw std::runtime_error("generic_sensor_policy_player steadytray variant was built without librealsense2; install librealsense2-dev and rebuild, or use --object-source udp/zero");
#elif !defined(ACTION_GO_WITH_APRILTAG)
    throw std::runtime_error("generic_sensor_policy_player steadytray variant was built without apriltag C library; install AprilRobotics apriltag and rebuild, or use --object-source udp/zero");
#else
    throw std::runtime_error("generic_sensor_policy_player steadytray variant AprilTag source unavailable");
#endif
  }
  ObjectPose latest() override { return {}; }
  std::string stats() const override { return "object_source=apriltag(unavailable)"; }
};
#endif

std::unique_ptr<ObjectSource> make_object_source(const Args& args) {
  if (args.object_source == "zero") return std::make_unique<ZeroObjectSource>();
  if (args.object_source == "udp") return std::make_unique<UdpObjectSource>(args.udp_port);
  if (args.object_source == "apriltag") return std::make_unique<AprilTagObjectSource>(args);
  throw std::runtime_error("unknown --object-source: " + args.object_source);
}

struct Config {
  float control_dt{0.02f};
  float ang_vel_scale{0.2f};
  float dof_pos_scale{1.0f};
  float dof_vel_scale{0.05f};
  std::array<float, 3> cmd_scale{1.f, 1.f, 1.f};
  std::vector<float> kp, kd;
  std::vector<float> default_angles_policy, action_scale_policy;
  std::vector<float> default_angles_robot, action_scale_robot;
  std::vector<int> policy_to_robot, robot_to_policy;
  std::vector<float> torque_limits, lower_limits, upper_limits;
  std::array<float, 3> cmd_init{0.5f, 0.f, 0.f};
};

Config load_config(const fs::path& path) {
  const YAML::Node y = YAML::LoadFile(path.string());
  Config c;
  c.control_dt = yamlc::required_scalar<float>(y, "policy.control_dt");
  c.ang_vel_scale = yamlc::required_scalar<float>(y, "observation.ang_vel_scale");
  c.dof_pos_scale = yamlc::required_scalar<float>(y, "observation.dof_pos_scale");
  c.dof_vel_scale = yamlc::required_scalar<float>(y, "observation.dof_vel_scale");
  auto cmd_scale = yamlc::required_vector_size<float>(y, "observation.cmd_scale", 3);
  c.cmd_scale = {cmd_scale[0], cmd_scale[1], cmd_scale[2]};
  auto cmd_init = yamlc::required_vector_size<float>(y, "motion.cmd_init", 3);
  c.cmd_init = {cmd_init[0], cmd_init[1], cmd_init[2]};
  yamlc::require_size("robot.motor_count", kNumJoints,
                      static_cast<size_t>(yamlc::required_scalar<int>(y, "robot.motor_count")));
  c.kp = yamlc::required_vector_size<float>(y, "robot.kp", kNumJoints);
  c.kd = yamlc::required_vector_size<float>(y, "robot.kd", kNumJoints);
  c.default_angles_policy = yamlc::required_vector_size<float>(y, "robot.default_angles", kNumJoints);
  c.action_scale_policy = yamlc::required_vector_size<float>(y, "robot.action_scale", kNumJoints);
  c.torque_limits = yamlc::required_vector_size<float>(y, "robot.torque_limits", kNumJoints);
  c.policy_to_robot = yamlc::required_vector_size<int>(y, "robot.joint_map", kNumJoints);
  c.robot_to_policy = yamlc::required_vector_size<int>(y, "robot.robot_to_policy", kNumJoints);
  yamlc::require_permutation("robot.joint_map", c.policy_to_robot, kNumJoints);
  yamlc::require_permutation("robot.robot_to_policy", c.robot_to_policy, kNumJoints);
  c.lower_limits = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", kNumJoints);
  c.upper_limits = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", kNumJoints);

  check_size("kp", kNumJoints, c.kp.size());
  check_size("kd", kNumJoints, c.kd.size());
  check_size("default_angles", kNumJoints, c.default_angles_policy.size());
  check_size("action_scale", kNumJoints, c.action_scale_policy.size());
  check_size("policy_to_robot", kNumJoints, c.policy_to_robot.size());
  check_size("robot_to_policy", kNumJoints, c.robot_to_policy.size());
  check_size("joint_limits_lower", kNumJoints, c.lower_limits.size());
  check_size("joint_limits_upper", kNumJoints, c.upper_limits.size());

  c.default_angles_robot.assign(kNumJoints, 0.f);
  c.action_scale_robot.assign(kNumJoints, 0.f);
  for (int robot = 0; robot < kNumJoints; ++robot) {
    const int policy = c.robot_to_policy[robot];
    if (policy < 0 || policy >= kNumJoints) throw std::runtime_error("robot_to_policy out of range");
    c.default_angles_robot[robot] = c.default_angles_policy[policy];
    c.action_scale_robot[robot] = c.action_scale_policy[policy];
  }
  return c;
}

class SteadyTrayPolicy {
 public:
  struct StateSnapshot {
    std::deque<std::vector<float>> frame_stack;
    std::deque<std::vector<float>> encoder_stack;
    std::vector<float> last_action_robot;
  };

  SteadyTrayPolicy(const Args& args, Config cfg)
      : args_(args), cfg_(std::move(cfg)), mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

  void load() {
    Ort::SessionOptions opt;
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    opt.SetIntraOpNumThreads(std::max(1, args_.onnx_threads));
    opt.SetInterOpNumThreads(1);
    session_ = Ort::Session(env_, args_.model.c_str(), opt);
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t nin = session_.GetInputCount();
    const size_t nout = session_.GetOutputCount();
    for (size_t i = 0; i < nin; ++i) {
      auto n = session_.GetInputNameAllocated(i, alloc);
      input_names_.emplace_back(n.get());
    }
    for (size_t i = 0; i < nout; ++i) {
      auto n = session_.GetOutputNameAllocated(i, alloc);
      output_names_.emplace_back(n.get());
    }
    if (input_names_.size() != 2 || output_names_.empty()) {
      throw std::runtime_error("SteadyTray ONNX must have two inputs and at least one output");
    }
    if (std::find(input_names_.begin(), input_names_.end(), "student_encoder_obs") == input_names_.end() ||
        std::find(input_names_.begin(), input_names_.end(), "policy_obs") == input_names_.end()) {
      throw std::runtime_error("SteadyTray ONNX missing required inputs: student_encoder_obs, policy_obs");
    }
    if (std::find(output_names_.begin(), output_names_.end(), "actions") == output_names_.end()) {
      throw std::runtime_error("SteadyTray ONNX missing required output: actions");
    }
    input_names_ = {"student_encoder_obs", "policy_obs"};
    output_names_ = {"actions"};
    std::cout << "[generic_sensor_policy_player][steadytray] loaded ONNX inputs=" << input_names_[0] << "," << input_names_[1]
              << " output=" << output_names_[0] << "\n";
  }

  void reset_histories() {
    frame_stack_.clear();
    encoder_stack_.clear();
    std::vector<float> zero_frame(kPropObsDim, 0.f);
    std::vector<float> zero_encoder(kStudentObsDim, 0.f);
    zero_encoder[96 + 3] = 1.f;  // identity object quaternion w
    for (int i = 0; i < kPolicyFrames; ++i) frame_stack_.push_back(zero_frame);
    for (int i = 0; i < kEncoderSeqLen; ++i) encoder_stack_.push_back(zero_encoder);
    last_action_robot_.assign(kNumJoints, 0.f);
  }

  std::vector<float> infer(const std::vector<float>& q_robot,
                           const std::vector<float>& dq_robot,
                           const std::array<float, 4>& quat_wxyz,
                           const std::array<float, 3>& gyro,
                           const std::array<float, 7>& object_obs) {
    if (q_robot.size() < kNumJoints || dq_robot.size() < kNumJoints) {
      throw std::runtime_error("robot state motor count too small");
    }
    std::vector<float> q_norm_robot(kNumJoints), dq_norm_robot(kNumJoints);
    for (int i = 0; i < kNumJoints; ++i) {
      q_norm_robot[i] = (q_robot[i] - cfg_.default_angles_robot[i]) * cfg_.dof_pos_scale;
      dq_norm_robot[i] = dq_robot[i] * cfg_.dof_vel_scale;
    }
    const auto grav = projected_gravity_from_quat_steadytray(quat_wxyz);
    std::array<float, 3> omega{gyro[0] * cfg_.ang_vel_scale, gyro[1] * cfg_.ang_vel_scale, gyro[2] * cfg_.ang_vel_scale};

    std::vector<float> q_policy(kNumJoints), dq_policy(kNumJoints), action_policy_prev(kNumJoints);
    for (int p = 0; p < kNumJoints; ++p) {
      const int r = cfg_.policy_to_robot[p];
      q_policy[p] = q_norm_robot[r];
      dq_policy[p] = dq_norm_robot[r];
      action_policy_prev[p] = last_action_robot_[r];
    }

    std::vector<float> frame(kPropObsDim, 0.f);
    fill_prop_frame(frame, omega, grav, q_policy, dq_policy, action_policy_prev);
    frame_stack_.push_back(frame);
    while (frame_stack_.size() > kPolicyFrames) frame_stack_.pop_front();

    std::vector<float> policy_obs = build_group_major_policy_obs();

    std::vector<float> enc(kStudentObsDim, 0.f);
    std::copy(frame.begin(), frame.end(), enc.begin());
    std::copy(object_obs.begin(), object_obs.end(), enc.begin() + kPropObsDim);
    encoder_stack_.push_back(enc);
    while (encoder_stack_.size() > kEncoderSeqLen) encoder_stack_.pop_front();

    std::vector<float> encoder_obs;
    encoder_obs.reserve(kEncoderSeqLen * kStudentObsDim);
    for (const auto& e : encoder_stack_) encoder_obs.insert(encoder_obs.end(), e.begin(), e.end());

    std::array<int64_t, 3> enc_shape{kEncoderSeqLen, 1, kStudentObsDim};
    std::array<int64_t, 2> pol_shape{1, kPolicyObsDim};
    Ort::Value enc_tensor = Ort::Value::CreateTensor<float>(mem_, encoder_obs.data(), encoder_obs.size(), enc_shape.data(), enc_shape.size());
    Ort::Value pol_tensor = Ort::Value::CreateTensor<float>(mem_, policy_obs.data(), policy_obs.size(), pol_shape.data(), pol_shape.size());
    std::array<Ort::Value, 2> inputs{std::move(enc_tensor), std::move(pol_tensor)};
    std::array<const char*, 2> in_names{input_names_[0].c_str(), input_names_[1].c_str()};
    const char* out_name = output_names_[0].c_str();
    auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), &out_name, 1);
    std::vector<float> action_policy = tensor_to_vec_exact(outs[0], kNumJoints, "SteadyTray actor actions");

    std::vector<float> action_robot(kNumJoints, 0.f);
    for (int pol = 0; pol < kNumJoints; ++pol) {
      const int robot = cfg_.policy_to_robot[pol];
      action_robot[robot] = action_policy[pol];
    }
    last_action_robot_ = action_robot;
    return action_robot;
  }

  std::vector<float> action_to_target(const std::vector<float>& action_robot) const {
    std::vector<float> target(kNumJoints, 0.f);
    for (int i = 0; i < kNumJoints; ++i) {
      target[i] = cfg_.default_angles_robot[i] + action_robot[i] * cfg_.action_scale_robot[i];
      target[i] = std::clamp(target[i], cfg_.lower_limits[i], cfg_.upper_limits[i]);
    }
    return target;
  }

  const Config& cfg() const { return cfg_; }
  StateSnapshot snapshot_state() const {
    return {frame_stack_, encoder_stack_, last_action_robot_};
  }
  void restore_state(const StateSnapshot& snapshot) {
    frame_stack_ = snapshot.frame_stack;
    encoder_stack_ = snapshot.encoder_stack;
    last_action_robot_ = snapshot.last_action_robot;
  }

 private:
  void fill_prop_frame(std::vector<float>& obs,
                       const std::array<float, 3>& omega,
                       const std::array<float, 3>& grav,
                       const std::vector<float>& q_policy,
                       const std::vector<float>& dq_policy,
                       const std::vector<float>& action_policy_prev) const {
    int idx = 0;
    for (int i = 0; i < 3; ++i) obs[idx++] = omega[i];
    for (int i = 0; i < 3; ++i) obs[idx++] = grav[i];
    for (int i = 0; i < 3; ++i) obs[idx++] = args_.cmd[i] * cfg_.cmd_scale[i];
    for (int i = 0; i < kNumJoints; ++i) obs[idx++] = q_policy[i];
    for (int i = 0; i < kNumJoints; ++i) obs[idx++] = dq_policy[i];
    for (int i = 0; i < kNumJoints; ++i) obs[idx++] = action_policy_prev[i];
  }

  std::vector<float> build_group_major_policy_obs() const {
    std::vector<float> out(kPolicyObsDim, 0.f);
    int idx = 0;
    auto copy_cols = [&](int start, int len) {
      for (const auto& f : frame_stack_) {
        for (int i = 0; i < len; ++i) out[idx++] = f[start + i];
      }
    };
    copy_cols(0, 3);
    copy_cols(3, 3);
    copy_cols(6, 3);
    copy_cols(9, kNumJoints);
    copy_cols(9 + kNumJoints, kNumJoints);
    copy_cols(9 + 2 * kNumJoints, kNumJoints);
    return out;
  }

  Args args_;
  Config cfg_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "generic_sensor_policy_player"};
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_;
  std::vector<std::string> input_names_, output_names_;
  std::deque<std::vector<float>> frame_stack_;
  std::deque<std::vector<float>> encoder_stack_;
  std::vector<float> last_action_robot_;
};

class UnitreeDDS {
 public:
  explicit UnitreeDDS(const Args& args,
                      const std::vector<float>& lower_limits,
                      const std::vector<float>& upper_limits)
      : args_(args), activation_blend_(args.activation_blend_sec) {
    event_log_.open_append(args_.event_log);
    CommandSafetyConfig safety_cfg;
    safety_cfg.lowstate_timeout_sec = 0.2;
    safety_filter_.set_config(safety_cfg);
    safety_limits_.lower = lower_limits;
    safety_limits_.upper = upper_limits;
    unitree::robot::ChannelFactory::Instance()->Init(0, args_.net);
    publisher_.reset(new unitree::robot::ChannelPublisher<LowCmd>(args_.cmd_topic));
    publisher_->InitChannel();
    subscriber_.reset(new unitree::robot::ChannelSubscriber<LowState>(args_.state_topic));
    subscriber_->InitChannel(std::bind(&UnitreeDDS::on_low_state, this, std::placeholders::_1), 1);
  }

  struct RobotState {
    std::array<float, 4> quat{1.f, 0.f, 0.f, 0.f};
    std::array<float, 3> gyro{0.f, 0.f, 0.f};
    std::vector<float> q;
    std::vector<float> dq;
    int mode_machine{0};
    uint64_t seq{0};
    std::chrono::steady_clock::time_point recv_stamp{};
  };

  void wait_for_state() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!connected_.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!connected_.load()) throw std::runtime_error("timeout waiting for lowstate");
    std::cout << "[generic_sensor_policy_player][steadytray] lowstate connected\n";
  }

  RobotState snapshot() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return state_;
  }

  void publish(const std::vector<float>& q, const std::vector<float>& kp, const std::vector<float>& kd,
               bool allow_stale_state = false, bool active_command = true,
               bool bypass_activation = false) {
    if (args_.dry_run || !publisher_) return;
    if (!active_command && args_.experimental_no_prestart_hold) {
      if (!prestart_hold_suppressed_logged_) {
        prestart_hold_suppressed_logged_ = true;
        event_log_.log("prestart_hold_suppressed",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("variant", "steadytray"),
                        event_string("reason", "experimental_handoff_ablation")});
      }
      return;
    }
    const auto s = snapshot();
    CommandSafetyState safety_state;
    safety_state.connected = connected_.load();
    safety_state.stamp = s.recv_stamp;
    safety_state.seq = s.seq;
    safety_state.q = s.q;
    JointCommand command{q, kp, kd};
    action_go::runtime::ActivationBlendResult blend_result;
    if (active_command) {
      blend_result = activation_blend_.apply(command, ActivationBlend::Clock::now(),
                                             bypass_activation);
      command = std::move(blend_result.command);
      if (blend_result.started) {
        event_log_.log("activation_blend_started",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("variant", "steadytray"),
                        event_number("duration_sec", args_.activation_blend_sec),
                        event_number("max_q_delta", blend_result.max_q_delta),
                        event_number("max_kp_delta", blend_result.max_kp_delta),
                        event_number("max_kd_delta", blend_result.max_kd_delta)});
      }
      if (bypass_activation && !activation_blend_bypass_logged_) {
        activation_blend_bypass_logged_ = true;
        event_log_.log("activation_blend_bypassed",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("variant", "steadytray"),
                        event_string("reason", "preview_commit")});
      }
    }
    action_go::runtime::FilteredCommand filtered;
    try {
      filtered = safety_filter_.filter(command.q, command.kp, command.kd,
                                       safety_state, safety_limits_, kNumJoints,
                                       allow_stale_state);
    } catch (const std::exception& e) {
      safety_failed_.store(true);
      event_log_.log("safety_stop",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray"),
                      event_string("reason", e.what()),
                      event_int("state_seq", static_cast<long long>(s.seq))});
      std::cerr << "[generic_sensor_policy_player][steadytray] safety stop: " << e.what() << "\n";
      g_stop_requested.store(true);
      return;
    }

    LowCmd cmd;
    cmd.mode_pr() = 0;
    cmd.mode_machine() = static_cast<uint8_t>(s.mode_machine > 0 ? s.mode_machine : args_.mode_machine);
    for (int i = 0; i < kNumJoints && i < static_cast<int>(cmd.motor_cmd().size()); ++i) {
      cmd.motor_cmd()[i].mode() = 1;
      cmd.motor_cmd()[i].q() = filtered.q[i];
      cmd.motor_cmd()[i].dq() = 0.f;
      cmd.motor_cmd()[i].tau() = 0.f;
      cmd.motor_cmd()[i].kp() = filtered.kp[i];
      cmd.motor_cmd()[i].kd() = filtered.kd[i];
    }
    cmd.crc() = crc32_core((uint32_t*)&cmd, (sizeof(cmd) >> 2) - 1);
    publisher_->Write(cmd);
    if (!active_command) {
      activation_blend_.observe_hold({filtered.q, filtered.kp, filtered.kd});
    } else if (blend_result.completed && !blend_result.bypassed &&
               args_.activation_blend_sec > 0.0) {
      event_log_.log("activation_blend_completed",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray"),
                      event_number("duration_sec", args_.activation_blend_sec)});
    }
    if (!first_command_logged_) {
      first_command_logged_ = true;
      event_log_.log("first_command_published",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray"),
                      event_string("command_phase", active_command ? "active" : "hold"),
                      event_int("state_seq", static_cast<long long>(s.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
    if (!active_command && !first_hold_logged_) {
      first_hold_logged_ = true;
      event_log_.log("first_hold_published",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray"),
                      event_int("state_seq", static_cast<long long>(s.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
    if (active_command && !first_active_command_logged_) {
      first_active_command_logged_ = true;
      event_log_.log("first_active_command",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray"),
                      event_int("state_seq", static_cast<long long>(s.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
  }

  void publish_hold_current(float kp_value = 20.f, float kd_value = 1.f) {
    const auto s = snapshot();
    if (s.q.size() < kNumJoints) return;
    publish(s.q, std::vector<float>(kNumJoints, kp_value),
            std::vector<float>(kNumJoints, kd_value), true, false);
  }

  void publish_damping() {
    if (args_.dry_run || !publisher_) return;
    const auto s = snapshot();
    if (s.q.size() < kNumJoints) return;
    std::vector<float> kp(kNumJoints, 0.f), kd(kNumJoints, 3.f);
    publish(s.q, kp, kd, true, false);
  }
  bool safety_failed() const { return safety_failed_.load(); }

 private:
  void on_low_state(const void* message) {
    const auto low = *(const LowState*)message;
    if (low.motor_state().size() < kNumJoints) return;
    const auto finite = [](float value) { return std::isfinite(value); };
    const auto quat = low.imu_state().quaternion();
    const auto gyro = low.imu_state().gyroscope();
    if (!std::all_of(quat.begin(), quat.end(), finite) ||
        !std::all_of(gyro.begin(), gyro.end(), finite)) return;
    std::lock_guard<std::mutex> lk(mtx_);
    state_.mode_machine = static_cast<int>(low.mode_machine());
    state_.recv_stamp = std::chrono::steady_clock::now();
    state_.seq = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    state_.quat = low.imu_state().quaternion();
    state_.gyro = low.imu_state().gyroscope();
    state_.q.assign(kNumJoints, 0.f);
    state_.dq.assign(kNumJoints, 0.f);
    for (size_t i = 0; i < kNumJoints; ++i) {
      state_.q[i] = low.motor_state()[i].q();
      state_.dq[i] = low.motor_state()[i].dq();
      if (!finite(state_.q[i]) || !finite(state_.dq[i])) return;
    }
    connected_.store(true);
  }

  const Args& args_;
  std::unique_ptr<unitree::robot::ChannelPublisher<LowCmd>> publisher_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  mutable std::mutex mtx_;
  RobotState state_;
  CommandSafetyFilter safety_filter_;
  CommandSafetyLimits safety_limits_;
  RuntimeEventLogger event_log_;
  ActivationBlend activation_blend_;
  bool first_command_logged_{false};
  bool first_hold_logged_{false};
  bool first_active_command_logged_{false};
  bool prestart_hold_suppressed_logged_{false};
  bool activation_blend_bypass_logged_{false};
  std::atomic<bool> safety_failed_{false};
  std::atomic<bool> connected_{false};
  std::atomic<uint64_t> sequence_{0};
};

void touch_file(const std::string& path) {
  if (path.empty()) return;
  fs::path p(path);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream out(path);
  out << "ready\n";
}

std::string hash_object_pose(const ObjectPose& pose) {
  uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const uint8_t*>(pose.obs.data());
  for (size_t i = 0; i < pose.obs.size() * sizeof(float); ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

void wait_for_start_file(const Args& args, UnitreeDDS* dds) {
  if (args.start_file.empty()) return;
  std::cout << "[generic_sensor_policy_player][steadytray] prestart ready; waiting for start file " << args.start_file << "\n";
  while (!g_stop_requested.load()) {
    if (fs::exists(args.start_file)) return;
    if (dds) dds->publish_hold_current();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void camera_check_loop(const Args& args) {
  auto source = make_object_source(args);
  const auto start = std::chrono::steady_clock::now();
  int count = 0;
  while (!g_stop_requested.load()) {
    const auto p = source->latest();
    std::cout << std::fixed << std::setprecision(3)
              << "[generic_sensor_policy_player][steadytray] camera-check valid=" << (p.valid ? 1 : 0)
              << " age=" << p.age_sec << " obs=[" << p.obs[0] << "," << p.obs[1] << "," << p.obs[2]
              << "," << p.obs[3] << "," << p.obs[4] << "," << p.obs[5] << "," << p.obs[6] << "] "
              << source->stats() << "\n";
    ++count;
    if (args.duration_sec > 0.0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= args.duration_sec) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  if (count == 0) throw std::runtime_error("camera-check produced no samples");
}

class SteadyTrayRunner {
 public:
  explicit SteadyTrayRunner(Args args) : args_(std::move(args)) {
    event_log_.open_append(args_.event_log);
    if (args_.params.empty()) throw std::runtime_error("--params is required");
    cfg_ = load_config(args_.params);
    if (!args_.cmd_overridden) args_.cmd = cfg_.cmd_init;
    policy_ = std::make_unique<SteadyTrayPolicy>(args_, cfg_);
  }

  void run() {
    if (!fs::exists(args_.model)) throw std::runtime_error("model not found: " + args_.model);
    if (!fs::exists(args_.params)) throw std::runtime_error("params not found: " + args_.params);
    policy_->load();
    policy_->reset_histories();
    if (args_.check_only) {
      event_log_.log("model_loaded",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray"),
                      event_string("model", args_.model)});
      event_log_.log("check_only_ok",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray")});
      std::cout << "[generic_sensor_policy_player][steadytray] check-only ok\n";
      return;
    }

    object_source_ = make_object_source(args_);
    if (args_.require_tag) wait_for_valid_object_pose();

    UnitreeDDS dds(args_, cfg_.lower_limits, cfg_.upper_limits);
    dds.wait_for_state();
    // Ready means a valid current-pose hold has already reached the DDS write
    // path.  Keep this ordering consistent with every other generic player.
    dds.publish_hold_current();
    if (!args_.ready_file.empty()) {
      touch_file(args_.ready_file);
      event_log_.log("ready_file_touched",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray"),
                      event_string("path", args_.ready_file)});
    }
    const bool preview_mode = !args_.preview_file.empty();
    std::vector<float> preview_target;
    if (preview_mode) {
      const auto policy_state = policy_->snapshot_state();
      const auto robot_state = dds.snapshot();
      ObjectPose object_snapshot = object_source_->latest();
      if (!object_snapshot.valid || object_snapshot.age_sec > args_.pose_timeout_sec) {
        throw std::runtime_error(
            "transactional preview requires a fresh object-pose snapshot: " +
            object_source_->stats());
      }
      const auto action = policy_->infer(
          robot_state.q, robot_state.dq, robot_state.quat,
          robot_state.gyro, object_snapshot.obs);
      preview_target = policy_->action_to_target(action);
      event_log_.log("preview_input_snapshot",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("template", "object_conditioned_actor"),
                      event_string("variant", "steadytray"),
                      event_string("source", args_.object_source),
                      event_int("input_sequence", static_cast<long long>(object_snapshot.sequence)),
                      event_number("input_freshness_sec", object_snapshot.age_sec),
                      event_string("input_hash", hash_object_pose(object_snapshot))});
      const auto decision = action_go::runtime::wait_preview_lifecycle(
          args_.preview_file, args_.commit_file, args_.discard_file,
          args_.preview_timeout_sec, std::max(args_.publish_hz, args_.hz),
          []() { return g_stop_requested.load(); },
          [&]() { dds.publish(preview_target, cfg_.kp, cfg_.kd, false, true, true); });
      if (decision != action_go::runtime::PreviewLifecycleDecision::kCommit) {
        policy_->restore_state(policy_state);
        event_log_.log("preview_state_restored",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("template", "object_conditioned_actor"),
                        event_string("variant", "steadytray")});
        if (decision == action_go::runtime::PreviewLifecycleDecision::kTimeout) {
          throw std::runtime_error("transactional preview timed out before commit");
        }
        return;
      }
      event_log_.log("preview_commit_first_command_consistent",
                     {event_string("player", "generic_sensor_policy_player"),
                      event_string("variant", "steadytray")});
    } else {
      wait_for_start_file(args_, &dds);
      if (!args_.start_file.empty() && fs::exists(args_.start_file)) {
        event_log_.log("start_file_seen",
                       {event_string("player", "generic_sensor_policy_player"),
                        event_string("variant", "steadytray"),
                        event_string("path", args_.start_file)});
      }
    }

    const auto start = std::chrono::steady_clock::now();
    auto next_infer = preview_mode
        ? std::chrono::steady_clock::now() +
              std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(1.0 / std::max(1, args_.hz)))
        : std::chrono::steady_clock::now();
    std::vector<float> target = preview_mode ? preview_target : dds.snapshot().q;
    if (target.size() < kNumJoints) target.assign(kNumJoints, 0.f);
    std::vector<float> from = target, to = target;
    auto interp_start = std::chrono::steady_clock::now();
    int infer_count = preview_mode ? 1 : 0;
    double infer_ms_avg = 0.0;
    double infer_ms_max = 0.0;
    long long policy_deadline_miss = 0;
    std::atomic<long long> publish_deadline_miss{0};
    std::atomic<long long> publish_count{0};
    std::atomic<double> max_publish_interval_ms{0.0};
    std::mutex command_mutex;
    bool command_active = preview_mode;
    std::atomic<bool> publisher_stop{false};
    std::exception_ptr publisher_error;
    std::mutex publisher_error_mutex;
    const auto publish_dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(args_.publish_hz, args_.hz)));
    std::thread publisher_thread([&]() {
      try {
        auto next_pub = std::chrono::steady_clock::now();
        auto previous_publish = std::chrono::steady_clock::time_point{};
        while (!publisher_stop.load() && !g_stop_requested.load()) {
          const auto now = std::chrono::steady_clock::now();
          std::vector<float> publish_target;
          bool publish_active = false;
          {
            std::lock_guard<std::mutex> lock(command_mutex);
            const double a = args_.target_interp_sec > 1e-6
                ? std::clamp(std::chrono::duration<double>(now - interp_start).count() /
                                 args_.target_interp_sec,
                             0.0, 1.0)
                : 1.0;
            publish_target = to;
            if (from.size() == to.size()) {
              for (size_t i = 0; i < publish_target.size(); ++i) {
                publish_target[i] = static_cast<float>(from[i] + a * (to[i] - from[i]));
              }
            }
            target = publish_target;
            publish_active = command_active;
          }
          dds.publish(publish_target, cfg_.kp, cfg_.kd, false, publish_active,
                      preview_mode);
          const auto published_at = std::chrono::steady_clock::now();
          if (previous_publish.time_since_epoch().count() != 0) {
            const double interval_ms =
                std::chrono::duration<double, std::milli>(published_at - previous_publish).count();
            double observed = max_publish_interval_ms.load();
            while (interval_ms > observed &&
                   !max_publish_interval_ms.compare_exchange_weak(observed, interval_ms)) {
            }
          }
          previous_publish = published_at;
          publish_count.fetch_add(1);
          next_pub += publish_dt;
          if (next_pub < published_at) {
            publish_deadline_miss.fetch_add(1);
            next_pub = published_at;
          }
          std::this_thread::sleep_until(next_pub);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(publisher_error_mutex);
        publisher_error = std::current_exception();
        publisher_stop.store(true);
      }
    });

    std::exception_ptr inference_error;
    try {
      while (!g_stop_requested.load() && !publisher_stop.load()) {
      const auto now = std::chrono::steady_clock::now();
      if (args_.duration_sec > 0.0 && std::chrono::duration<double>(now - start).count() >= args_.duration_sec) break;

      if (now >= next_infer) {
        const auto infer_begin = std::chrono::steady_clock::now();
        const auto s = dds.snapshot();
        if (s.q.size() < kNumJoints || s.dq.size() < kNumJoints) throw std::runtime_error("lowstate motor count too small");
        ObjectPose op = object_source_->latest();
        if (!op.valid || op.age_sec > args_.pose_timeout_sec) {
          if (args_.require_tag || args_.exit_on_tag_lost) {
            throw std::runtime_error("object pose missing/stale: " + object_source_->stats());
          }
        }
        std::vector<float> next_target;
        bool next_active = true;
        if (!preview_mode && infer_count < args_.warmup_steps) {
          next_target = s.q;
          next_active = false;
        } else {
          const auto action = policy_->infer(s.q, s.dq, s.quat, s.gyro, op.obs);
          next_target = policy_->action_to_target(action);
        }
        {
          std::lock_guard<std::mutex> lock(command_mutex);
          from = target.size() == to.size() ? target : to;
          to = std::move(next_target);
          interp_start = now;
          command_active = next_active;
        }
        ++infer_count;
        const auto infer_end = std::chrono::steady_clock::now();
        const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
        infer_ms_avg += (infer_ms - infer_ms_avg) / static_cast<double>(infer_count);
        infer_ms_max = std::max(infer_ms_max, infer_ms);
        if (args_.stats_interval > 0 && infer_count % args_.stats_interval == 0) {
          const float target0 = to.empty() ? 0.f : to.front();
          std::cout << "[generic_sensor_policy_player][steadytray] infer=" << infer_count << " cmd=(" << args_.cmd[0] << "," << args_.cmd[1] << "," << args_.cmd[2]
                    << ") target0=" << target0 << " " << object_source_->stats() << "\n";
          event_log_.log("runtime_metrics",
                         {event_string("player", "generic_sensor_policy_player"),
                          event_string("template", "object_conditioned_actor"),
                          event_string("variant", "steadytray"),
                          event_int("step", infer_count),
                          event_number("infer_ms_last", infer_ms),
                          event_number("infer_ms_avg", infer_ms_avg),
                          event_number("infer_ms_max", infer_ms_max),
                          event_number("object_age_sec", std::isfinite(op.age_sec) ? op.age_sec : -1.0),
                          event_number("target0", target0),
                          event_int("policy_deadline_miss", policy_deadline_miss),
                          event_int("publish_count", publish_count.load()),
                          event_int("publish_deadline_miss", publish_deadline_miss.load()),
                          event_number("max_publish_interval_ms",
                                       max_publish_interval_ms.load())});
          infer_ms_max = 0.0;
        }
        next_infer += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / std::max(1, args_.hz)));
        const auto after_infer = std::chrono::steady_clock::now();
        if (next_infer < after_infer) { ++policy_deadline_miss; next_infer = after_infer; }
      }

        if (next_infer > std::chrono::steady_clock::now()) {
          std::this_thread::sleep_until(next_infer);
        }
      }
    } catch (...) {
      inference_error = std::current_exception();
    }
    publisher_stop.store(true);
    if (publisher_thread.joinable()) publisher_thread.join();
    {
      std::lock_guard<std::mutex> lock(publisher_error_mutex);
      if (publisher_error) std::rethrow_exception(publisher_error);
    }
    if (inference_error) std::rethrow_exception(inference_error);
    event_log_.log("runtime_metrics",
                   {event_string("player", "generic_sensor_policy_player"),
                    event_string("template", "object_conditioned_actor"),
                    event_string("variant", "steadytray"),
                    event_int("publish_count", publish_count.load()),
                    event_int("policy_deadline_miss", policy_deadline_miss),
                    event_int("publish_deadline_miss", publish_deadline_miss.load()),
                    event_number("max_publish_interval_ms",
                                 max_publish_interval_ms.load())});
    if (!args_.no_damping_on_stop) dds.publish_damping();
    if (dds.safety_failed()) throw std::runtime_error("CommandSafetyFilter stopped SteadyTray publishing");
    std::cout << "[generic_sensor_policy_player][steadytray] stopped\n";
  }

 private:
  void wait_for_valid_object_pose() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(args_.camera_timeout_sec));
    while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
      const auto p = object_source_->latest();
      if (p.valid && p.age_sec <= args_.pose_timeout_sec) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("timeout waiting for valid object pose: " + object_source_->stats());
  }

  Args args_;
  Config cfg_;
  std::unique_ptr<SteadyTrayPolicy> policy_;
  std::unique_ptr<ObjectSource> object_source_;
  RuntimeEventLogger event_log_;
};

}  // namespace

int run_steadytray_runtime(int argc, char** argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    Args args = parse_args(argc, argv);
    if (args.camera_check) {
      camera_check_loop(args);
      return 0;
    }
    SteadyTrayRunner runner(std::move(args));
    runner.run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[generic_sensor_policy_player][steadytray] ERROR: " << e.what() << std::endl;
    return 1;
  }
}
