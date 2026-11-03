#include "action_go/motion_action_controller.h"
#include "action_go/runtime/network_command.h"
#include "action_go/runtime/bridge_gate.h"
#include "action_go/runtime/bridge_risk_monitor.h"
#include "action_go/runtime/runtime_session.h"

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;
using action_go::runtime::AuthenticatedNetworkCommandParser;
using action_go::runtime::BoundedNetworkCommandQueue;
using action_go::runtime::NetworkCommand;
using action_go::runtime::NetworkCommandType;

namespace {
std::atomic<int> g_stop_level{0};

void normal_exit_signal_handler(int) {
  int expected = g_stop_level.load();
  // First Ctrl+C: normal exit to WALKRUN.
  // Second Ctrl+C: stronger damping exit.
  if (expected >= 1) {
    g_stop_level.store(2);
    return;
  }
  while (expected < 1 && !g_stop_level.compare_exchange_weak(expected, 1)) {}
}

void damping_signal_handler(int) { g_stop_level.store(2); }

enum RemoteButton : int {
  kR1 = 0, kL1 = 1, kStart = 2, kSelect = 3,
  kR2 = 4, kL2 = 5, kF1 = 6, kF2 = 7,
  kA = 8, kB = 9, kX = 10, kY = 11,
  kUp = 12, kRight = 13, kDown = 14, kLeft = 15
};

const char* buttonName(int button) {
  static const char* names[] = {
      "R1", "L1", "Start", "Select", "R2", "L2", "F1", "F2",
      "A", "B", "X", "Y", "Up", "Right", "Down", "Left"};
  return (button >= 0 && button < 16) ? names[button] : "Unknown";
}

std::string trim(const std::string& input) {
  size_t begin = 0;
  while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) ++begin;
  size_t end = input.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) --end;
  return input.substr(begin, end - begin);
}

std::string unquote(std::string value) {
  value = trim(value);
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

bool parseBoolText(std::string value, bool default_value) {
  (void)default_value;
  value = unquote(value);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "true" || value == "1" || value == "yes" || value == "on") return true;
  if (value == "false" || value == "0" || value == "no" || value == "off") return false;
  throw std::runtime_error("invalid boolean value: " + value);
}

int parseButton(std::string name) {
  name = unquote(name);
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (name == "r1" || name == "rb") return kR1;
  if (name == "l1" || name == "lb") return kL1;
  if (name == "start") return kStart;
  if (name == "select" || name == "back") return kSelect;
  if (name == "r2" || name == "rt") return kR2;
  if (name == "l2" || name == "lt") return kL2;
  if (name == "f1") return kF1;
  if (name == "f2") return kF2;
  if (name == "a") return kA;
  if (name == "b") return kB;
  if (name == "x") return kX;
  if (name == "y") return kY;
  if (name == "up") return kUp;
  if (name == "right") return kRight;
  if (name == "down") return kDown;
  if (name == "left") return kLeft;
  throw std::runtime_error("unknown remote button: " + name);
}

uint16_t bit(int button) { return static_cast<uint16_t>(1u << button); }

uint16_t parseCombo(const std::string& text) {
  uint16_t mask = 0;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, '+')) {
    item = unquote(item);
    if (!item.empty()) mask |= bit(parseButton(item));
  }
  if (mask == 0) throw std::runtime_error("empty remote combo: " + text);
  return mask;
}

uint16_t parseOptionalCombo(std::string text) {
  text = unquote(trim(text));
  std::string lower = text;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower.empty() || lower == "none" || lower == "null" || lower == "off" ||
      lower == "disable" || lower == "disabled" || lower == "-") {
    return 0;
  }
  return parseCombo(text);
}

std::string comboName(uint16_t mask) {
  std::string out;
  for (int i = 0; i < 16; ++i) {
    if (mask & bit(i)) {
      if (!out.empty()) out += "+";
      out += buttonName(i);
    }
  }
  return out.empty() ? "None" : out;
}

bool comboRising(uint16_t now, uint16_t prev, uint16_t combo) {
  return (now & combo) == combo && (prev & combo) != combo;
}

constexpr double kPi = 3.141592653589793238462643383279502884;

std::array<float, 4> normalizeQuatWxyz(std::array<float, 4> q) {
  if (!std::all_of(q.begin(), q.end(), [](float value) { return std::isfinite(value); })) {
    return {0.f, 0.f, 0.f, 0.f};
  }
  const double n = std::sqrt(static_cast<double>(q[0]) * q[0] +
                             static_cast<double>(q[1]) * q[1] +
                             static_cast<double>(q[2]) * q[2] +
                             static_cast<double>(q[3]) * q[3]);
  if (!(n > 1e-6) || !std::isfinite(n)) return {0.f, 0.f, 0.f, 0.f};
  for (float& v : q) v = static_cast<float>(v / n);
  return q;
}

std::array<double, 3> projectedGravityFromQuatWxyz(const std::array<float, 4>& quat_wxyz) {
  // Same criterion as RoboJuDo safety_check(): compute body-frame gravity and
  // trigger on arccos(-gravity_z). Unitree LowState quaternion is [w,x,y,z].
  const auto q = normalizeQuatWxyz(quat_wxyz);
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double r20 = 2.0 * (x * z - y * w);
  const double r21 = 2.0 * (y * z + x * w);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);
  return {-r20, -r21, -r22};
}

double tiltDegFromQuatWxyz(const std::array<float, 4>& quat_wxyz) {
  if (!std::all_of(quat_wxyz.begin(), quat_wxyz.end(), [](float value) { return std::isfinite(value); })) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double norm_sq = static_cast<double>(quat_wxyz[0]) * quat_wxyz[0] +
                         static_cast<double>(quat_wxyz[1]) * quat_wxyz[1] +
                         static_cast<double>(quat_wxyz[2]) * quat_wxyz[2] +
                         static_cast<double>(quat_wxyz[3]) * quat_wxyz[3];
  if (!(norm_sq > 1e-12) || !std::isfinite(norm_sq)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto gravity = projectedGravityFromQuatWxyz(quat_wxyz);
  if (!std::isfinite(gravity[2])) return std::numeric_limits<double>::quiet_NaN();
  const double c = std::clamp(-gravity[2], -1.0, 1.0);
  return std::acos(c) * 180.0 / kPi;
}

struct ImuSample {
  bool valid{false};
  double tilt_deg{0.0};
  std::array<float, 4> quat_wxyz{1.f, 0.f, 0.f, 0.f};
  std::chrono::steady_clock::time_point stamp{};
};

class RemoteMonitor {
 public:
  explicit RemoteMonitor(const std::string& topic = "rt/lowstate") {
    subscriber_.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(topic));
    subscriber_->InitChannel(std::bind(&RemoteMonitor::onLowState, this, std::placeholders::_1), 1);
  }

  bool connected() const { return connected_.load(); }

  uint16_t mask() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mask_;
  }

  ImuSample imu() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ImuSample{has_imu_, tilt_deg_, quat_wxyz_, imu_stamp_};
  }

  action_go::runtime::BridgeRiskState riskState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return risk_state_;
  }

 private:
  void onLowState(const void* message) {
    const auto* low = static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
    const auto quat = low->imu_state().quaternion();
    const double tilt_deg = tiltDegFromQuatWxyz(quat);
    const auto stamp = std::chrono::steady_clock::now();
    const auto gyro = low->imu_state().gyroscope();

    const auto& remote = low->wireless_remote();

    {
      std::lock_guard<std::mutex> lock(mutex_);
      quat_wxyz_ = quat;
      tilt_deg_ = tilt_deg;
      imu_stamp_ = stamp;
      has_imu_ = std::isfinite(tilt_deg) &&
                 std::all_of(quat.begin(), quat.end(), [](float value) { return std::isfinite(value); });
      risk_state_.stamp = stamp;
      risk_state_.gyro = gyro;
      risk_state_.projected_gravity = {static_cast<float>(projectedGravityFromQuatWxyz(quat)[0]),
                                       static_cast<float>(projectedGravityFromQuatWxyz(quat)[1]),
                                       static_cast<float>(projectedGravityFromQuatWxyz(quat)[2])};
      if (low->motor_state().size() >= risk_state_.q.size()) {
        for (size_t i = 0; i < risk_state_.q.size(); ++i) {
          risk_state_.q[i] = low->motor_state()[i].q();
          risk_state_.dq[i] = low->motor_state()[i].dq();
        }
      }

      if (remote.size() >= 4) {
        // RoboJuDo / Unitree mapping:
        // bit0..15 = R1,L1,Start,Select,R2,L2,F1,F2,A,B,X,Y,Up,Right,Down,Left
        mask_ = static_cast<uint16_t>(remote[2]) |
                (static_cast<uint16_t>(remote[3]) << 8);
      }
    }
    connected_.store(has_imu_);
  }

  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> subscriber_;
  mutable std::mutex mutex_;
  uint16_t mask_{0};
  std::array<float, 4> quat_wxyz_{1.f, 0.f, 0.f, 0.f};
  double tilt_deg_{0.0};
  std::chrono::steady_clock::time_point imu_stamp_{};
  bool has_imu_{false};
  action_go::runtime::BridgeRiskState risk_state_{};
  std::atomic<bool> connected_{false};
};

class UdpCommandReceiver {
 public:
  UdpCommandReceiver(std::string bind_addr, int port, std::string multicast_group,
                     AuthenticatedNetworkCommandParser::Config auth_config,
                     bool require_authentication)
      : bind_addr_(std::move(bind_addr)),
        multicast_group_(std::move(multicast_group)),
        port_(port),
        parser_(std::move(auth_config)),
        require_authentication_(require_authentication) {}

  ~UdpCommandReceiver() { stop(); }

  void start() {
    if (running_.load()) return;

    sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ < 0) throw std::runtime_error("failed to create udp socket");

    int reuse = 1;
    ::setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (bind_addr_.empty() || bind_addr_ == "0.0.0.0") {
      addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, bind_addr_.c_str(), &addr.sin_addr) != 1) {
      throw std::runtime_error("bad udp bind address: " + bind_addr_);
    }

    if (::bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      const std::string err = "failed to bind udp " + bind_addr_ + ":" + std::to_string(port_);
      ::close(sock_);
      sock_ = -1;
      throw std::runtime_error(err);
    }

    if (!multicast_group_.empty()) {
      ip_mreq mreq{};
      if (::inet_pton(AF_INET, multicast_group_.c_str(), &mreq.imr_multiaddr) != 1) {
        throw std::runtime_error("bad multicast group: " + multicast_group_);
      }
      mreq.imr_interface.s_addr = htonl(INADDR_ANY);
      if (::setsockopt(sock_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0) {
        throw std::runtime_error("failed to join multicast group: " + multicast_group_);
      }
    }

    running_.store(true);
    thread_ = std::thread(&UdpCommandReceiver::threadMain, this);
  }

  void stop() {
    running_.store(false);
    if (sock_ >= 0) {
      ::shutdown(sock_, SHUT_RDWR);
    }
    if (thread_.joinable()) thread_.join();
    if (sock_ >= 0) {
      ::close(sock_);
      sock_ = -1;
    }
  }

  bool pop(NetworkCommand& cmd) {
    return queue_.pop(cmd);
  }

 private:
  void threadMain() {
    while (running_.load()) {
      fd_set readfds;
      FD_ZERO(&readfds);
      FD_SET(sock_, &readfds);
      timeval tv{};
      tv.tv_sec = 0;
      tv.tv_usec = 100000;
      const int ret = ::select(sock_ + 1, &readfds, nullptr, nullptr, &tv);
      if (ret <= 0) continue;

      char buf[1024];
      sockaddr_in peer{};
      socklen_t peer_len = sizeof(peer);
      const ssize_t n = ::recvfrom(sock_, buf, sizeof(buf) - 1, 0,
                                   reinterpret_cast<sockaddr*>(&peer), &peer_len);
      if (n <= 0) continue;
      buf[n] = '\0';

      try {
        const int64_t now_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        NetworkCommand cmd = parser_.parse(buf, now_epoch_ms, require_authentication_);
        char peer_ip[INET_ADDRSTRLEN] = {0};
        ::inet_ntop(AF_INET, &peer.sin_addr, peer_ip, sizeof(peer_ip));
        std::cout << "[action_go_remote_test] udp command from "
                  << peer_ip << ":" << ntohs(peer.sin_port)
                  << " command='" << cmd.raw << "'" << std::endl;
        if (!queue_.push(std::move(cmd))) {
          std::cerr << "[action_go_remote_test] ignored UDP action: command queue full\n";
        }
      } catch (const std::exception& e) {
        std::cerr << "[action_go_remote_test] ignored bad udp command: "
                  << e.what() << std::endl;
      }
    }
  }

  std::string bind_addr_;
  std::string multicast_group_;
  int port_{45678};
  int sock_{-1};
  std::atomic<bool> running_{false};
  std::thread thread_;
  BoundedNetworkCommandQueue queue_{64};
  AuthenticatedNetworkCommandParser parser_;
  bool require_authentication_{false};
};

struct Binding {
  uint16_t combo{static_cast<uint16_t>(bit(kR2) | bit(kDown))};
  std::string action{"kick_right"};
};

struct SafetyConfig {
  bool enable{true};
  bool auto_recovery{false};
  std::string action{"ufo_stand"};
  std::string return_mode{"walkrun"};
  // RoboJuDo uses 1.0 rad (~57 deg) as the fallen threshold.
  double fall_angle_deg{57.3};
  double recover_angle_deg{20.0};
  int fall_hold_ms{60};
  int recover_hold_ms{1200};
  int cooldown_ms{1000};
};

struct RemoteConfig {
  std::vector<Binding> bindings;
  uint16_t emergency_combo{0};
  uint16_t walkrun_exit_combo{0};
  uint16_t damping_exit_combo{0};
  uint16_t shell_start_combo{bit(kStart)};
  uint16_t shell_start_alt_combo{static_cast<uint16_t>(bit(kLeft) | bit(kB))};
  std::string shell_start_action{"bfm_zero_stand"};
  std::string shell_start_alt_action{"ufo_stand"};
  uint16_t shell_damping_combo{static_cast<uint16_t>(bit(kSelect) | bit(kStart))};
  uint16_t shell_walkrun_combo{static_cast<uint16_t>(bit(kR2) | bit(kA))};
  uint16_t lie_down_combo{static_cast<uint16_t>(bit(kL2) | bit(kDown))};
  std::string lie_action{"lie_down_beyondmimic"};
  SafetyConfig safety;
};

struct Args {
  std::string net{"enP8p1s0"};
  std::string config;
  std::string remote_map;
  std::string event_log;
  double startup_delay_sec{0.0};
  bool udp_listen{false};
  bool dry_run{false};
  std::string udp_bind{"127.0.0.1"};
  std::string udp_group;
  std::string udp_auth_key_file;
  std::string udp_auth_key_id;
  int udp_port{45678};
  std::string risk_model;
  std::string b_action{"walkrun"};
  int auto_switch_step{-1};
  bool bridge_enabled{true};
  RemoteConfig remote;
};

Binding parseBinding(const std::string& text) {
  const auto pos = text.find_first_of(":=");
  if (pos == std::string::npos || pos == 0 || pos + 1 >= text.size()) {
    throw std::runtime_error("bad binding format, expected Combo:action, got: " + text);
  }
  Binding b;
  b.combo = parseCombo(text.substr(0, pos));
  b.action = unquote(text.substr(pos + 1));
  if (b.action.empty()) throw std::runtime_error("empty action in binding: " + text);
  return b;
}

RemoteConfig loadRemoteConfigFile(const std::string& path) {
  RemoteConfig cfg;
  cfg.bindings.clear();

  std::ifstream input(path);
  if (!input.good()) throw std::runtime_error("failed to open remote map: " + path);

  bool in_bindings = false;
  std::string line;
  int line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    if (trim(line).empty()) continue;

    const auto indent = line.find_first_not_of(' ');
    const std::string stripped = trim(line);
    if (stripped == "bindings:") {
      in_bindings = true;
      continue;
    }

    const auto colon = stripped.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("bad remote map line " + std::to_string(line_no) + ": " + stripped);
    }
    const std::string key = unquote(stripped.substr(0, colon));
    const std::string value = unquote(stripped.substr(colon + 1));

    if (in_bindings && indent >= 2) {
      cfg.bindings.push_back(Binding{parseCombo(key), value});
      continue;
    }

    in_bindings = false;
    if (key == "emergency" || key == "emergency_combo") {
      cfg.emergency_combo = parseOptionalCombo(value);
    } else if (key == "normal_exit" || key == "walkrun_exit" ||
               key == "exit" || key == "walkrun_exit_combo") {
      cfg.walkrun_exit_combo = parseOptionalCombo(value);
    } else if (key == "damping_exit" || key == "damping" || key == "damping_exit_combo") {
      cfg.damping_exit_combo = parseOptionalCombo(value);
    } else if (key == "shell_start" || key == "shell_start_combo" ||
               key == "arm" || key == "arm_combo") {
      cfg.shell_start_combo = parseOptionalCombo(value);
    } else if (key == "shell_start_alt" || key == "shell_start_alt_combo" ||
               key == "arm_alt" || key == "arm_alt_combo") {
      cfg.shell_start_alt_combo = parseOptionalCombo(value);
    } else if (key == "shell_start_action" || key == "arm_action") {
      cfg.shell_start_action = value;
    } else if (key == "shell_start_alt_action" || key == "arm_alt_action") {
      cfg.shell_start_alt_action = value;
    } else if (key == "shell_damping" || key == "shell_damping_combo" ||
               key == "shell_stop" || key == "shell_stop_combo") {
      cfg.shell_damping_combo = parseOptionalCombo(value);
    } else if (key == "shell_walkrun" || key == "shell_walkrun_combo" ||
               key == "shell_walkrun_stop" || key == "walkrun_shell" ||
               key == "walkrun_shell_combo") {
      cfg.shell_walkrun_combo = parseOptionalCombo(value);
    } else if (key == "lie_down" || key == "lie_down_combo") {
      cfg.lie_down_combo = parseCombo(value);
    } else if (key == "lie_action" || key == "lie_down_action") {
      cfg.lie_action = value;
    } else if (key == "safety" || key == "safety_enable") {
      cfg.safety.enable = parseBoolText(value, cfg.safety.enable);
    } else if (key == "safety_auto_recovery" || key == "auto_recovery") {
      cfg.safety.auto_recovery = parseBoolText(value, cfg.safety.auto_recovery);
    } else if (key == "safety_action" || key == "recovery_action") {
      cfg.safety.action = value;
    } else if (key == "safety_return_mode" || key == "recovery_return_mode") {
      cfg.safety.return_mode = value;
    } else if (key == "fall_angle_deg" || key == "safety_fall_angle_deg") {
      cfg.safety.fall_angle_deg = std::stod(value);
    } else if (key == "recover_angle_deg" || key == "safety_recover_angle_deg") {
      cfg.safety.recover_angle_deg = std::stod(value);
    } else if (key == "fall_hold_ms" || key == "safety_fall_hold_ms") {
      cfg.safety.fall_hold_ms = std::stoi(value);
    } else if (key == "recover_hold_ms" || key == "safety_recover_hold_ms") {
      cfg.safety.recover_hold_ms = std::stoi(value);
    } else if (key == "safety_cooldown_ms" || key == "recovery_cooldown_ms") {
      cfg.safety.cooldown_ms = std::stoi(value);
    } else {
      throw std::runtime_error("unknown remote map key at line " + std::to_string(line_no) + ": " + key);
    }
  }

  if (cfg.bindings.empty()) throw std::runtime_error("remote map contains no bindings: " + path);
  return cfg;
}

void appendDefaultMotionConfigPath(std::string& config) {
  if (!config.empty()) return;
  const fs::path cwd = fs::current_path();
  const fs::path c1 = cwd / "config/motion_actions.yaml";
  const fs::path c2 = cwd / "../action_go/config/motion_actions.yaml";
  const fs::path c3 = cwd / "../../action_go/config/motion_actions.yaml";
  for (const auto& p : {c1, c2, c3}) {
    if (fs::exists(p)) {
      config = p.lexically_normal().string();
      return;
    }
  }
  config = c1.lexically_normal().string();
}

void appendDefaultRemoteMapPath(std::string& remote_map) {
  if (!remote_map.empty()) return;
  const fs::path cwd = fs::current_path();
  const fs::path c1 = cwd / "config/remote_actions.yaml";
  const fs::path c2 = cwd / "../action_go/config/remote_actions.yaml";
  const fs::path c3 = cwd / "../../action_go/config/remote_actions.yaml";
  for (const auto& p : {c1, c2, c3}) {
    if (fs::exists(p)) {
      remote_map = p.lexically_normal().string();
      return;
    }
  }
}

Args parseArgs(int argc, char** argv) {
  Args args;
  std::vector<std::string> cli_bindings;
  bool cli_safety_enable_set = false;
  bool cli_safety_enable = args.remote.safety.enable;
  bool cli_auto_recovery_set = false;
  bool cli_auto_recovery = args.remote.safety.auto_recovery;
  std::string cli_safety_action;
  std::string cli_safety_return_mode;
  std::string cli_shell_start_combo;
  std::string cli_shell_start_alt_combo;
  std::string cli_shell_damping_combo;
  std::string cli_shell_walkrun_combo;
  std::string cli_lie_down_combo;
  std::string cli_lie_action;
  double cli_fall_angle_deg = -1.0;
  double cli_recover_angle_deg = -1.0;
  int cli_fall_hold_ms = -1;
  int cli_recover_hold_ms = -1;
  int cli_safety_cooldown_ms = -1;

  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (k == "--net") args.net = need("--net");
    else if (k == "--config") args.config = need("--config");
    else if (k == "--remote-map") args.remote_map = need("--remote-map");
    else if (k == "--event-log") args.event_log = need("--event-log");
    else if (k == "--dry-run") args.dry_run = true;
    else if (k == "--startup-delay-sec" || k == "--start-delay-sec") args.startup_delay_sec = std::stod(need("--startup-delay-sec"));
    else if (k == "--bind") cli_bindings.push_back(need("--bind"));
    else if (k == "--udp-listen" || k == "--upper-listen") args.udp_listen = true;
    else if (k == "--udp-bind") args.udp_bind = need("--udp-bind");
    else if (k == "--udp-port") args.udp_port = std::stoi(need("--udp-port"));
    else if (k == "--udp-group" || k == "--multicast-group") args.udp_group = need("--udp-group");
    else if (k == "--udp-auth-key-file") args.udp_auth_key_file = need("--udp-auth-key-file");
    else if (k == "--udp-auth-key-id") args.udp_auth_key_id = need("--udp-auth-key-id");
    else if (k == "--risk-model") args.risk_model = need("--risk-model");
    else if (k == "--b-action") args.b_action = need("--b-action");
    else if (k == "--auto-switch-step") args.auto_switch_step = std::stoi(need("--auto-switch-step"));
    else if (k == "--bridge-disable") args.bridge_enabled = false;
    else if (k == "--safety-enable") { cli_safety_enable_set = true; cli_safety_enable = true; }
    else if (k == "--safety-disable") { cli_safety_enable_set = true; cli_safety_enable = false; }
    else if (k == "--safety-auto-recovery") { cli_auto_recovery_set = true; cli_auto_recovery = true; }
    else if (k == "--safety-no-auto-recovery") { cli_auto_recovery_set = true; cli_auto_recovery = false; }
    else if (k == "--safety-action") cli_safety_action = need("--safety-action");
    else if (k == "--safety-return-mode") cli_safety_return_mode = need("--safety-return-mode");
    else if (k == "--shell-start-combo" || k == "--arm-combo") cli_shell_start_combo = need(k.c_str());
    else if (k == "--shell-start-alt-combo" || k == "--arm-alt-combo") cli_shell_start_alt_combo = need(k.c_str());
    else if (k == "--shell-damping-combo" || k == "--shell-stop-combo") cli_shell_damping_combo = need(k.c_str());
    else if (k == "--shell-walkrun-combo" || k == "--walkrun-shell-combo") cli_shell_walkrun_combo = need(k.c_str());
    else if (k == "--lie-down-combo") cli_lie_down_combo = need("--lie-down-combo");
    else if (k == "--lie-action" || k == "--lie-down-action") cli_lie_action = need(k.c_str());
    else if (k == "--fall-angle-deg" || k == "--safety-fall-angle-deg") cli_fall_angle_deg = std::stod(need("--fall-angle-deg"));
    else if (k == "--recover-angle-deg" || k == "--safety-recover-angle-deg") cli_recover_angle_deg = std::stod(need("--recover-angle-deg"));
    else if (k == "--fall-hold-ms" || k == "--safety-fall-hold-ms") cli_fall_hold_ms = std::stoi(need("--fall-hold-ms"));
    else if (k == "--recover-hold-ms" || k == "--safety-recover-hold-ms") cli_recover_hold_ms = std::stoi(need("--recover-hold-ms"));
    else if (k == "--safety-cooldown-ms") cli_safety_cooldown_ms = std::stoi(need("--safety-cooldown-ms"));
    else if (k == "--help") {
      std::cout
          << "Usage:\n"
          << "  action_go_remote_test --net enP8p1s0 --config config/motion_actions.yaml --remote-map config/remote_actions.yaml\n"
          << "  action_go_remote_test --udp-listen --udp-port 45678\n"
          << "  action_go_remote_test --udp-listen --udp-bind 0.0.0.0 --udp-auth-key-file /run/secrets/actiongo-udp.key\n"
          << "  action_go_remote_test --bind R2+Down:kick_right --bind Left+X:hlqk\n\n"
          << "Remote map YAML:\n"
          << "  bindings:\n"
          << "    R2+Down: kick_right\n"
          << "    Left+X: hlqk\n"
          << "  emergency: disabled\n"
          << "  normal_exit: disabled\n"
          << "  damping_exit: disabled\n"
          << "  shell_start: Start\n"
          << "  shell_start_alt: Left+B\n"
          << "  shell_start_action: bfm_zero_stand\n"
          << "  shell_start_alt_action: ufo_stand\n"
          << "  shell_damping: Select+Start\n"
          << "  shell_walkrun: R2+A\n"
          << "  lie_down: L2+Down\n"
          << "  lie_action: lie_down_beyondmimic\n"
          << "  safety_enable: true\n"
          << "  safety_auto_recovery: false\n"
          << "  safety_action: ufo_stand\n"
          << "  fall_angle_deg: 57.3       # RoboJuDo 1.0 rad\n"
          << "  recover_angle_deg: 30.0\n"
          << "  recover_hold_ms: 1200\n\n"
          << "Safety options:\n"
          << "  --safety-enable/--safety-disable\n"
          << "  --safety-auto-recovery/--safety-no-auto-recovery\n"
          << "  --safety-action ufo_stand --safety-return-mode walkrun\n"
          << "  --fall-angle-deg 57.3 --recover-angle-deg 30 --recover-hold-ms 1200\n\n"
          << "Shell options:\n"
          << "  --shell-start-combo Start --shell-start-alt-combo Left+B\n"
          << "  --shell-damping-combo Select+Start --shell-walkrun-combo R2+A\n"
          << "  --lie-down-combo L2+Down --lie-action lie_down_beyondmimic\n\n"
          << "Startup options:\n"
          << "  --startup-delay-sec 10     wait after remote ready before main loop/safety\n\n"
          << "Experiment logging:\n"
          << "  --event-log /tmp/actiongo_supervisor.jsonl  write supervisor JSONL events\n\n"
          << "Dry-run mode:\n"
          << "  --dry-run                  do not StopMove/UserCtrl/switch official; run players with no-LowCmd dry-run flags\n\n"
          << "Keyboard fallback:\n"
          << "  Ctrl+C once         -> normal exit: switch to WALKRUN and exit\n"
          << "  Ctrl+C twice/Ctrl+\\ -> dampingStop() and exit\n\n"
          << "Upper computer UDP commands:\n"
          << "  ACTION kick_right\n"
          << "  ACTION kick_right delay_ms=500\n"
          << "  action=kick_right start_ms=1730000000000\n"
          << "  emergency | normal_exit | damping_exit\n"
          << "Legacy text is accepted only on loopback. Non-loopback/group control requires signed JSON.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }

  appendDefaultMotionConfigPath(args.config);
  appendDefaultRemoteMapPath(args.remote_map);
  if (args.risk_model.empty()) {
    if (const char* model = std::getenv("ACTION_GO_RISK_MODEL")) args.risk_model = model;
  }
  if (args.event_log.empty()) {
    if (const char* env_log = std::getenv("ACTION_GO_EVENT_LOG")) {
      args.event_log = env_log;
    }
  }

  if (!args.remote_map.empty()) {
    args.remote = loadRemoteConfigFile(args.remote_map);
  }

  if (cli_safety_enable_set) args.remote.safety.enable = cli_safety_enable;
  if (cli_auto_recovery_set) args.remote.safety.auto_recovery = cli_auto_recovery;
  if (!cli_safety_action.empty()) args.remote.safety.action = cli_safety_action;
  if (!cli_safety_return_mode.empty()) args.remote.safety.return_mode = cli_safety_return_mode;
  if (!cli_shell_start_combo.empty()) args.remote.shell_start_combo = parseOptionalCombo(cli_shell_start_combo);
  if (!cli_shell_start_alt_combo.empty()) args.remote.shell_start_alt_combo = parseOptionalCombo(cli_shell_start_alt_combo);
  if (!cli_shell_damping_combo.empty()) args.remote.shell_damping_combo = parseOptionalCombo(cli_shell_damping_combo);
  if (!cli_shell_walkrun_combo.empty()) args.remote.shell_walkrun_combo = parseOptionalCombo(cli_shell_walkrun_combo);
  if (!cli_lie_down_combo.empty()) args.remote.lie_down_combo = parseCombo(cli_lie_down_combo);
  if (!cli_lie_action.empty()) args.remote.lie_action = cli_lie_action;
  if (cli_fall_angle_deg >= 0.0) args.remote.safety.fall_angle_deg = cli_fall_angle_deg;
  if (cli_recover_angle_deg >= 0.0) args.remote.safety.recover_angle_deg = cli_recover_angle_deg;
  if (cli_fall_hold_ms >= 0) args.remote.safety.fall_hold_ms = cli_fall_hold_ms;
  if (cli_recover_hold_ms >= 0) args.remote.safety.recover_hold_ms = cli_recover_hold_ms;
  if (cli_safety_cooldown_ms >= 0) args.remote.safety.cooldown_ms = cli_safety_cooldown_ms;

  const auto& safety = args.remote.safety;
  if (!std::isfinite(safety.fall_angle_deg) || !std::isfinite(safety.recover_angle_deg) ||
      safety.fall_angle_deg <= 0.0 || safety.fall_angle_deg > 180.0 ||
      safety.recover_angle_deg < 0.0 || safety.recover_angle_deg >= safety.fall_angle_deg ||
      safety.fall_hold_ms <= 0 || safety.recover_hold_ms <= 0 || safety.cooldown_ms < 0) {
    throw std::runtime_error("remote safety thresholds must be finite, ordered, and positive");
  }

  for (const auto& b : cli_bindings) {
    args.remote.bindings.push_back(parseBinding(b));
  }
  if (args.remote.bindings.empty()) {
    args.remote.bindings.push_back(Binding{static_cast<uint16_t>(bit(kR2) | bit(kDown)), "kick_right"});
  }
  return args;
}

} // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, normal_exit_signal_handler);
  std::signal(SIGTERM, normal_exit_signal_handler);
  std::signal(SIGQUIT, damping_signal_handler);

  try {
    Args args = parseArgs(argc, argv);
    if (args.udp_listen && args.udp_bind.empty()) {
      throw std::runtime_error("--udp-bind must be explicit; use 127.0.0.1 for local control");
    }
    const bool require_udp_auth = args.udp_listen &&
        (!action_go::runtime::is_loopback_bind_address(args.udp_bind) || !args.udp_group.empty());
    if (require_udp_auth && args.udp_auth_key_file.empty()) {
      throw std::runtime_error(
          "non-loopback or multicast UDP control requires --udp-auth-key-file");
    }
    action_go::runtime::RuntimeSession runtime_session(
        args.event_log, args.config, args.remote_map, args.net, args.dry_run);
    args.event_log = runtime_session.event_path();
    try {
    AuthenticatedNetworkCommandParser::Config udp_auth_config;
    if (args.udp_listen && !args.udp_auth_key_file.empty()) {
      udp_auth_config.key = action_go::runtime::load_udp_auth_key_file(args.udp_auth_key_file);
      udp_auth_config.key_id = args.udp_auth_key_id;
    }
    std::cout << "[action_go_remote_test] net=" << args.net
              << " config=" << args.config;
    if (!args.remote_map.empty()) std::cout << " remote_map=" << args.remote_map;
    if (args.udp_listen) {
      std::cout << " udp=" << args.udp_bind << ":" << args.udp_port;
      if (!args.udp_group.empty()) std::cout << " group=" << args.udp_group;
    }
    if (args.dry_run) std::cout << " DRY_RUN=1";
    if (!args.event_log.empty()) std::cout << " event_log=" << args.event_log;
    std::cout << std::endl;

    unitree::robot::ChannelFactory::Instance()->Init(0, args.net);

    unitree::robot::g1::LocoClient loco;
    loco.Init();
    loco.SetTimeout(10.0f);

    MotionActionController controller;
    controller.setEventLogPath(args.event_log);
    if (!controller.init(&loco, args.config, args.net, args.dry_run)) {
      runtime_session.notice("ERROR", "Initialization failed: " + controller.lastError());
      runtime_session.result("initialization_failed");
      std::cerr << "[action_go_remote_test] init failed: " << controller.lastError() << std::endl;
      return 1;
    }

    std::unique_ptr<UdpCommandReceiver> udp_receiver;
    if (args.udp_listen) {
      udp_receiver = std::make_unique<UdpCommandReceiver>(
          args.udp_bind, args.udp_port, args.udp_group, std::move(udp_auth_config), require_udp_auth);
      udp_receiver->start();
      std::cout << "[action_go_remote_test] upper-computer UDP listen enabled on "
                << args.udp_bind << ":" << args.udp_port;
      if (!args.udp_group.empty()) std::cout << " multicast_group=" << args.udp_group;
      std::cout << std::endl;
    }

    RemoteMonitor remote;
    if (args.dry_run) {
      std::cout << "[action_go_remote_test][dry-run] enabled: actions will NOT call StopMove, "
                << "will NOT switch UserCtrl, and will NOT request official walkrun/passive. "
                << "Players are run with no-LowCmd dry-run flags." << std::endl;
    }
    std::cout << "[action_go_remote_test] waiting for lowstate/IMU/remote..." << std::endl;
    while (!remote.connected()) {
      if (g_stop_level.load() != 0) {
        std::cerr << "[action_go_remote_test] exit requested while waiting for lowstate/IMU/remote"
                  << std::endl;
        return g_stop_level.load() >= 2 ? 3 : 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::cout << "[action_go_remote_test] remote ready. Bindings:" << std::endl;
    for (const auto& binding : args.remote.bindings) {
      std::cout << "  " << comboName(binding.combo) << " -> " << binding.action << std::endl;
    }
    auto print_special_combo = [](uint16_t combo, const std::string& desc) {
      if (combo == 0) {
        std::cout << "  disabled -> " << desc << "\n";
      } else {
        std::cout << "  " << comboName(combo) << " -> " << desc << "\n";
      }
    };
    print_special_combo(args.remote.emergency_combo, "dampingStop, return to shell");
    print_special_combo(args.remote.walkrun_exit_combo, "switch to WALKRUN, exit");
    print_special_combo(args.remote.damping_exit_combo, "dampingStop, exit");
    print_special_combo(args.remote.shell_start_combo, "ARM shell: enable actions/safety");
    print_special_combo(args.remote.shell_start_alt_combo, "ARM shell: enable actions/safety (alternate)");
    std::cout << "  primary ARM recovery -> " << args.remote.shell_start_action << "\n";
    std::cout << "  alternate ARM recovery -> " << args.remote.shell_start_alt_action << "\n";
    print_special_combo(args.remote.shell_damping_combo, "dampingStop, return to shell");
    print_special_combo(args.remote.shell_walkrun_combo, "switch to WALKRUN, return to shell");
    print_special_combo(args.remote.lie_down_combo, args.remote.lie_action + ", then passive shell");
    if (args.remote.safety.enable) {
      std::cout << "  IMU safety -> " << args.remote.safety.action
                << " when tilt >= " << args.remote.safety.fall_angle_deg << " deg for "
                << args.remote.safety.fall_hold_ms << " ms; return "
                << args.remote.safety.return_mode << " when tilt <= "
                << args.remote.safety.recover_angle_deg << " deg for "
                << args.remote.safety.recover_hold_ms << " ms; auto recovery="
                << (args.remote.safety.auto_recovery ? "on" : "off") << "\n";
    } else {
      std::cout << "  IMU safety disabled\n";
    }
    std::cout << "[action_go_remote_test][shell] initial state=SHELL_IDLE: "
              << "no action/safety takeover until "
              << comboName(args.remote.shell_start_combo) << " or "
              << comboName(args.remote.shell_start_alt_combo) << " is pressed\n";
    runtime_session.notice("INFO", "Remote ready; waiting for ARM");
    runtime_session.render(controller.controlState(), "", "");

    if (args.startup_delay_sec > 0.0) {
      std::cout << "[action_go_remote_test] startup delay "
                << args.startup_delay_sec
                << " sec before enabling remote actions/safety..." << std::endl;
      const auto delay_until = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(args.startup_delay_sec));
      while (std::chrono::steady_clock::now() < delay_until) {
        if (g_stop_level.load() != 0) {
          std::cerr << "[action_go_remote_test] interrupted during startup delay" << std::endl;
          return 130;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      std::cout << "[action_go_remote_test] startup delay done, main loop enabled" << std::endl;
    }

    uint16_t prev_mask = remote.mask();
    MotionActionController::Status last_status = controller.update();
    std::string active_action;
    std::string scheduled_action;
    std::chrono::steady_clock::time_point scheduled_time;

    enum class SafetyState { Normal, Recovering, Damped };
    SafetyState safety_state = SafetyState::Normal;
    bool fall_timer_active = false;
    bool recover_timer_active = false;
    auto fall_since = std::chrono::steady_clock::now();
    auto recover_since = std::chrono::steady_clock::now();
    auto last_safety_finish = std::chrono::steady_clock::now() -
        std::chrono::milliseconds(std::max(0, args.remote.safety.cooldown_ms));
    bool shell_armed = false;
    bool lie_action_active = false;
    std::unique_ptr<action_go::runtime::BridgeRiskMonitor> bridge_risk;
    action_go::runtime::BridgeGate bridge_gate;
    action_go::runtime::DoubleClickCombo normal_bridge_combo;
    action_go::runtime::DoubleClickCombo emergency_bridge_combo;
    int bridge_step = 0;
    bool bridge_normal_pending = false;
    bool bridge_emergency_pending = false;
    if (args.bridge_enabled && !args.risk_model.empty()) {
      bridge_risk = std::make_unique<action_go::runtime::BridgeRiskMonitor>(
          action_go::runtime::BridgeRiskMonitor::Config{args.risk_model});
      bridge_risk->load();
      std::cout << "[action_go_remote_test][bridge] risk model loaded: " << args.risk_model
                << " provider=" << bridge_risk->risk_percent() << " B=" << args.b_action << std::endl;
    }
    std::string selected_safety_action = args.remote.safety.action;
    bool logging_failure_reported = false;

    auto requestAction = [&](const std::string& action,
                             std::chrono::steady_clock::time_point when,
                             const std::string& source) {
      const auto now = std::chrono::steady_clock::now();
      if (safety_state == SafetyState::Recovering && action != selected_safety_action) {
        std::cerr << "[action_go_remote_test][safety] ignore action while recovering from "
                  << source << ": " << action << std::endl;
        return;
      }
      if (when > now) {
        scheduled_action = action;
        scheduled_time = when;
        const auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(when - now).count();
        std::cout << "[action_go_remote_test] scheduled action from " << source
                  << ": " << action << " in " << wait_ms << " ms" << std::endl;
        return;
      }
      if (controller.isRunning()) {
        std::cerr << "[action_go_remote_test] ignore action while running from "
                  << source << ": " << action << std::endl;
        return;
      }
      std::cout << "[action_go_remote_test] start action from "
                << source << ": " << action << std::endl;
      if (!controller.startAction(action)) {
        std::cerr << "[action_go_remote_test] startAction failed: "
                  << controller.lastError() << std::endl;
      } else {
        active_action = action;
        runtime_session.notice("INFO", "Starting " + action);
        last_status = MotionActionController::Status::SwitchingToUserCtrl;
      }
    };

    auto ensureSafetyRecoveryRunning = [&](const std::string& source) -> bool {
      const std::string current = controller.currentAction();
      if (controller.isRunning() && current == selected_safety_action) {
        return true;
      }
      if (controller.isRunning()) {
        std::cerr << "[action_go_remote_test][safety] interrupt current action '"
                  << current << "' before recovery (source=" << source << ")" << std::endl;
        controller.interruptForRecovery();
      }
      std::cout << "[action_go_remote_test][safety] start recovery action: "
                << selected_safety_action << std::endl;
      if (!controller.startAction(selected_safety_action)) {
        std::cerr << "[action_go_remote_test][safety] start recovery failed: "
                  << controller.lastError() << std::endl;
        return false;
      }
      active_action = selected_safety_action;
      runtime_session.notice("WARN", "Safety recovery: " + selected_safety_action);
      last_status = MotionActionController::Status::SwitchingToUserCtrl;
      return true;
    };

    auto startSafetyRecovery = [&](double tilt_deg) {
      scheduled_action.clear();
      fall_timer_active = false;
      recover_timer_active = false;
      std::cerr << "[action_go_remote_test][safety] robot fallen: tilt="
                << tilt_deg << " deg >= " << args.remote.safety.fall_angle_deg
                << " deg, damping" << std::endl;
      safety_state = SafetyState::Damped;
      shell_armed = false;
      active_action.clear();
      controller.dampingStop();
      runtime_session.notice("WARN", "Robot fall detected; requesting DAMPING");
      if (args.remote.safety.auto_recovery) {
        shell_armed = true;
        safety_state = SafetyState::Recovering;
        std::cerr << "[action_go_remote_test][safety] auto recovery enabled, switch to "
                  << selected_safety_action << std::endl;
        ensureSafetyRecoveryRunning("fall_detected");
      }
    };

    auto safetyTick = [&]() {
      if (!args.remote.safety.enable) return;
      if (safety_state == SafetyState::Damped) return;

      const auto now = std::chrono::steady_clock::now();
      const ImuSample imu = remote.imu();

      if (safety_state == SafetyState::Recovering) {
        // BFM/UFO owns recovery once started.  Do not apply the normal
        // fall-angle safety path here: posture is expected to remain above
        // fall_angle_deg during much of a get-up motion.  Keep only payload
        // failure protection and the stable-upright handoff to walkrun.
        const std::string current = controller.currentAction();
        if (controller.hasError() || !controller.isRunning() ||
            current != selected_safety_action) {
          std::cerr << "[action_go_remote_test][safety] recovery failed; "
                    << "enter damping and return to SHELL_IDLE: "
                    << controller.lastError() << std::endl;
          scheduled_action.clear();
          active_action.clear();
          shell_armed = false;
          safety_state = SafetyState::Damped;
          recover_timer_active = false;
          controller.dampingStop();
          last_status = controller.update();
          return;
        }

        if (!imu.valid || now - imu.stamp > std::chrono::milliseconds(500)) {
          recover_timer_active = false;
          return;
        }
        if (imu.tilt_deg > args.remote.safety.recover_angle_deg) {
          recover_timer_active = false;
          return;
        }
        if (!recover_timer_active) {
          recover_since = now;
          recover_timer_active = true;
        }
        if (now - recover_since <
            std::chrono::milliseconds(std::max(0, args.remote.safety.recover_hold_ms))) {
          return;
        }

        std::cout << "[action_go_remote_test][safety] recovered upright: tilt="
                  << imu.tilt_deg << " deg <= " << args.remote.safety.recover_angle_deg
                  << " deg, switch back to " << args.remote.safety.return_mode << std::endl;
        if (controller.stopAndSwitchToOfficial(args.remote.safety.return_mode)) {
          active_action.clear();
          scheduled_action.clear();
          safety_state = SafetyState::Normal;
          recover_timer_active = false;
          last_safety_finish = now;
          last_status = controller.update();
        } else {
          std::cerr << "[action_go_remote_test][safety] switch back warning: "
                    << controller.lastError() << std::endl;
          recover_timer_active = false;
        }
        return;
      }

      if (controller.isRunning()) {
        const std::string current = controller.currentAction();
        if (!current.empty() && !controller.actionSafetyEnabled(current)) {
          fall_timer_active = false;
          recover_timer_active = false;
          return;
        }
      }

      if (!imu.valid) return;
      if (now - imu.stamp > std::chrono::milliseconds(500)) return;

      const bool fallen = imu.tilt_deg >= args.remote.safety.fall_angle_deg;
      recover_timer_active = false;
      if (now - last_safety_finish <
          std::chrono::milliseconds(std::max(0, args.remote.safety.cooldown_ms))) {
        fall_timer_active = false;
        return;
      }
      if (!fallen) {
        fall_timer_active = false;
        return;
      }
      if (!fall_timer_active) {
        fall_since = now;
        fall_timer_active = true;
      }
      if (now - fall_since >=
          std::chrono::milliseconds(std::max(0, args.remote.safety.fall_hold_ms))) {
        startSafetyRecovery(imu.tilt_deg);
      }
    };

    auto resetSafetyState = [&]() {
      safety_state = SafetyState::Normal;
      fall_timer_active = false;
      recover_timer_active = false;
      last_safety_finish = std::chrono::steady_clock::now() -
          std::chrono::milliseconds(std::max(0, args.remote.safety.cooldown_ms));
    };

    auto enterShellDamping = [&](const std::string& source) {
      std::cerr << "[action_go_remote_test][shell] " << source
                << ": dampingStop and return to SHELL_IDLE (program stays alive)"
                << std::endl;
      scheduled_action.clear();
      active_action.clear();
      lie_action_active = false;
      shell_armed = false;
      resetSafetyState();
      controller.dampingStop();
      g_stop_level.store(0);
      last_status = controller.update();
    };

    auto switchToBridgeB = [&]() {
      if (args.b_action == "walkrun") return controller.stopAndSwitchToOfficial("walkrun");
      if (!controller.hasAction(args.b_action)) {
        std::cerr << "[action_go_remote_test][bridge] B action is not registered: " << args.b_action << std::endl;
        return false;
      }
      return controller.startAction(args.b_action);
    };

    auto bridgeTick = [&]() {
      if (!bridge_risk) return;
      const float risk = bridge_risk->update(remote.riskState());
      const uint16_t mask = remote.mask();
      const bool normal_request = shell_armed &&
          normal_bridge_combo.update((mask & bit(kL2)) != 0, (mask & bit(kB)) != 0);
      const bool emergency_request = shell_armed &&
          emergency_bridge_combo.update((mask & bit(kL2)) != 0, (mask & bit(kY)) != 0);
      if (normal_request) bridge_normal_pending = true;
      if (emergency_request) bridge_emergency_pending = true;
      ++bridge_step;
      const bool auto_normal = args.auto_switch_step >= 0 && bridge_step == args.auto_switch_step;
      const auto imu = remote.imu();
      const auto state = remote.riskState();
      double dq_rms = 0.0;
      for (float value : state.dq) dq_rms += static_cast<double>(value) * value;
      dq_rms = std::sqrt(dq_rms / state.dq.size());
      double gyro_norm = 0.0;
      for (float value : state.gyro) gyro_norm += static_cast<double>(value) * value;
      const bool stand_ready = imu.valid && imu.tilt_deg <= 20.0 &&
                               std::sqrt(gyro_norm) <= 0.8 && dq_rms <= 1.5;
      const auto output = bridge_gate.tick(risk,
                                           bridge_normal_pending || (shell_armed && auto_normal),
                                           bridge_emergency_pending, stand_ready);
      if (output.command == action_go::runtime::BridgeCommand::kStartBfm ||
          output.command == action_go::runtime::BridgeCommand::kDamping ||
          output.rejected) {
        bridge_normal_pending = false;
        bridge_emergency_pending = false;
      }
      if (output.rejected) {
        std::cerr << "[action_go_remote_test][bridge] normal switch rejected risk=" << risk
                  << " level=" << output.risk_level << std::endl;
      }
      if (output.command == action_go::runtime::BridgeCommand::kStartBfm) {
        if (controller.isRunning()) controller.interruptForRecovery();
        if (!controller.startAction("bfm_zero_stand")) {
          std::cerr << "[action_go_remote_test][bridge] BFM start failed: " << controller.lastError() << std::endl;
          bridge_gate.reset();
          controller.dampingStop();
        } else {
          active_action = "bfm_zero_stand";
          std::cout << "[action_go_remote_test][bridge] risk=" << risk << " -> BFM stand" << std::endl;
        }
      } else if (output.command == action_go::runtime::BridgeCommand::kDamping) {
        std::cerr << "[action_go_remote_test][bridge] emergency observation timeout risk=" << risk << std::endl;
        controller.dampingStop();
      } else if (output.command == action_go::runtime::BridgeCommand::kSwitchToB) {
        if (!switchToBridgeB()) {
          std::cerr << "[action_go_remote_test][bridge] B switch failed: " << controller.lastError() << std::endl;
          controller.dampingStop();
        } else {
          active_action.clear();
          std::cout << "[action_go_remote_test][bridge] BFM ready -> B=" << args.b_action << std::endl;
        }
      }
    };

    auto enterShellWalkrun = [&](const std::string& source) {
      std::cerr << "[action_go_remote_test][shell] " << source
                << ": switch to WALKRUN and return to SHELL_IDLE (program stays alive)"
                << std::endl;
      scheduled_action.clear();
      active_action.clear();
      lie_action_active = false;
      shell_armed = false;
      resetSafetyState();
      if (!controller.stopAndSwitchToOfficial("walkrun")) {
        std::cerr << "[action_go_remote_test][shell] walkrun switch warning: "
                  << controller.lastError() << std::endl;
      }
      else runtime_session.notice("INFO", "WALKRUN confirmed");
      g_stop_level.store(0);
      last_status = controller.update();
    };

    auto armShell = [&](const std::string& source, const std::string& recovery_action) {
      if (controller.isRunning()) {
        std::cerr << "[action_go_remote_test][shell] cannot arm while action is running: "
                  << controller.currentAction() << std::endl;
        return;
      }
      shell_armed = true;
      selected_safety_action = recovery_action.empty() ? args.remote.safety.action : recovery_action;
      scheduled_action.clear();
      resetSafetyState();
      const ImuSample imu = remote.imu();
      std::cout << "[action_go_remote_test][shell] ARMED by " << source
                << ": actions and IMU safety enabled, recovery="
                << selected_safety_action << std::endl;
      runtime_session.notice("INFO", "ARMED by " + source);
      fall_timer_active = false;
      recover_timer_active = false;
      if (imu.valid && imu.tilt_deg >= args.remote.safety.recover_angle_deg) {
        safety_state = SafetyState::Recovering;
        std::cout << "[action_go_remote_test][shell] tilt=" << imu.tilt_deg
                  << " deg, starting selected recovery action" << std::endl;
        ensureSafetyRecoveryRunning("arm:" + source);
      }
    };

    auto startLieAction = [&](const std::string& source) {
      std::cerr << "[action_go_remote_test][shell] lie-down requested by "
                << source << ": action=" << args.remote.lie_action
                << " (will return to passive shell)" << std::endl;
      scheduled_action.clear();
      resetSafetyState();
      shell_armed = false;  // Do not let IMU safety interrupt the lie-down action.
      lie_action_active = true;
      if (controller.isRunning()) {
        std::cerr << "[action_go_remote_test][shell] interrupt current action '"
                  << controller.currentAction() << "' before lie-down" << std::endl;
        controller.interruptForRecovery();
      }
      if (!controller.startAction(args.remote.lie_action)) {
        std::cerr << "[action_go_remote_test][shell] lie-down start failed: "
                  << controller.lastError() << std::endl;
        lie_action_active = false;
        controller.dampingStop();
        last_status = controller.update();
        return;
      }
      active_action = args.remote.lie_action;
      last_status = MotionActionController::Status::SwitchingToUserCtrl;
    };

    auto handleNetworkCommand = [&](const NetworkCommand& cmd) -> bool {
      if (cmd.type == NetworkCommandType::kEmergency) {
        enterShellDamping("udp stop");
        return false;
      }
      if (cmd.type == NetworkCommandType::kDampingExit) {
        std::cerr << "[action_go_remote_test] udp damping exit requested" << std::endl;
        controller.dampingStop();
        return true;
      }
      if (cmd.type == NetworkCommandType::kWalkrunExit) {
        if (!shell_armed) {
          std::cerr << "[action_go_remote_test] ignore udp walkrun_exit while unarmed" << std::endl;
          return false;
        }
        std::cerr << "[action_go_remote_test] udp normal exit requested" << std::endl;
        if (!controller.stopAndSwitchToOfficial("walkrun")) {
          std::cerr << "[action_go_remote_test] official switch failed; keep supervisor active: "
                    << controller.lastError() << std::endl;
          return false;
        }
        return true;
      }
      if (!shell_armed) {
        std::cerr << "[action_go_remote_test][shell] ignore udp action while SHELL_IDLE: "
                  << cmd.action << " (press "
                  << comboName(args.remote.shell_start_combo)
                  << " to arm)" << std::endl;
        return false;
      }
      if (!controller.hasAction(cmd.action)) {
        std::cerr << "[action_go_remote_test] ignore UDP request for unregistered action: "
                  << cmd.action << std::endl;
        return false;
      }

      auto when = std::chrono::steady_clock::now();
      if (cmd.execute_at_ms > 0) {
        const auto now_sys = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const int64_t wait_ms = cmd.execute_at_ms - now_sys;
        if (wait_ms > 0) {
          when += std::chrono::milliseconds(wait_ms);
        }
      } else if (cmd.delay_ms > 0) {
        when += std::chrono::milliseconds(cmd.delay_ms);
      }
      requestAction(cmd.action, when, "udp");
      return false;
    };

    while (true) {
      const uint16_t now_mask = remote.mask();
      const int stop_level = g_stop_level.load();

      if (udp_receiver) {
        NetworkCommand cmd;
        while (udp_receiver->pop(cmd)) {
          if (handleNetworkCommand(cmd)) return 0;
        }
      }

      if (stop_level >= 2 || comboRising(now_mask, prev_mask, args.remote.damping_exit_combo)) {
        std::cerr << "[action_go_remote_test] damping stop requested, exiting" << std::endl;
        controller.dampingStop();
        return 3;
      }
      if (stop_level == 1 || comboRising(now_mask, prev_mask, args.remote.walkrun_exit_combo)) {
        std::cerr << "[action_go_remote_test] normal exit requested, switching to WALKRUN" << std::endl;
        if (!controller.stopAndSwitchToOfficial("walkrun")) {
          std::cerr << "[action_go_remote_test] normal exit warning: "
                    << controller.lastError() << std::endl;
          return 2;
        }
        return 0;
      }
      if (comboRising(now_mask, prev_mask, args.remote.shell_walkrun_combo)) {
        enterShellWalkrun("remote " + comboName(args.remote.shell_walkrun_combo));
        prev_mask = now_mask;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      if (comboRising(now_mask, prev_mask, args.remote.shell_damping_combo)) {
        enterShellDamping("remote " + comboName(args.remote.shell_damping_combo));
        prev_mask = now_mask;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      if (comboRising(now_mask, prev_mask, args.remote.shell_start_combo)) {
        armShell("remote " + comboName(args.remote.shell_start_combo),
                 args.remote.shell_start_action);
      } else if (comboRising(now_mask, prev_mask, args.remote.shell_start_alt_combo)) {
        armShell("remote " + comboName(args.remote.shell_start_alt_combo),
                 args.remote.shell_start_alt_action);
      }
      if (comboRising(now_mask, prev_mask, args.remote.lie_down_combo)) {
        startLieAction("remote " + comboName(args.remote.lie_down_combo));
      }
      if (comboRising(now_mask, prev_mask, args.remote.emergency_combo)) {
        enterShellDamping("remote " + comboName(args.remote.emergency_combo));
      }

      if (shell_armed) {
        safetyTick();
      }
      bridgeTick();

      if (shell_armed && !scheduled_action.empty() && std::chrono::steady_clock::now() >= scheduled_time) {
        const std::string action = scheduled_action;
        scheduled_action.clear();
        requestAction(action, std::chrono::steady_clock::now(), "scheduled");
      }

      if (shell_armed && !controller.isRunning()) {
        for (const auto& binding : args.remote.bindings) {
          if (comboRising(now_mask, prev_mask, binding.combo)) {
            requestAction(binding.action, std::chrono::steady_clock::now(), "remote");
            break;
          }
        }
      }

      const auto status = controller.update();
      if ((!controller.logsHealthy() || controller.droppedLogRecords()) && !logging_failure_reported) {
        logging_failure_reported = true;
        runtime_session.notice("ERROR", "Supervisor event logging is incomplete");
      }
      std::string displayed_state = controller.controlState();
      if (status == MotionActionController::Status::SwitchingToUserCtrl) displayed_state = "SWITCHING_TO_USER_CTRL";
      else if (status == MotionActionController::Status::SwitchingBack) displayed_state += " -> OFFICIAL";
      else if (status == MotionActionController::Status::SwitchBackBlocked) displayed_state = "SWITCH_BACK_BLOCKED";
      const std::string timing_path = active_action.empty() ? "" : controller.currentActionEventLog();
      const auto status_imu = remote.imu();
      runtime_session.render(args.dry_run ? "DRY_RUN" : displayed_state, active_action, timing_path,
          !status_imu.valid || std::chrono::steady_clock::now() - status_imu.stamp > std::chrono::seconds(1));
      if (status != last_status) {
        std::cout << "[action_go_remote_test] status="
                  << MotionActionController::statusToString(status) << std::endl;
        if (status == MotionActionController::Status::Done) {
          const std::string finished_action = active_action;
          runtime_session.notice("INFO", "Completed " + finished_action);
          std::cout << "[action_go_remote_test] action done: " << finished_action << std::endl;
          active_action.clear();
          if (lie_action_active || finished_action == args.remote.lie_action) {
            lie_action_active = false;
            shell_armed = false;
            resetSafetyState();
            std::cout << "[action_go_remote_test][shell] lie-down done, ensure PASSIVE and stay in SHELL_IDLE"
                      << std::endl;
            if (!controller.stopAndSwitchToOfficial("passive")) {
              std::cerr << "[action_go_remote_test][shell] passive switch warning: "
                        << controller.lastError() << std::endl;
            }
            last_status = controller.update();
            prev_mask = now_mask;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
          }
        } else if (status == MotionActionController::Status::Failed) {
          runtime_session.notice("ERROR", active_action + ": " + controller.lastError());
          std::cerr << "[action_go_remote_test] action failed: "
                    << controller.lastError() << std::endl;
          const std::string failed_action = active_action;
          if (lie_action_active || active_action == args.remote.lie_action) {
            lie_action_active = false;
            shell_armed = false;
            resetSafetyState();
            controller.dampingStop();
            last_status = controller.update();
            active_action.clear();
            prev_mask = now_mask;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
          }
          active_action.clear();
          if (failed_action == "bfm_zero_stand") {
            bridge_gate.reset();
            controller.dampingStop();
          }
        }
        last_status = status;
      }

      prev_mask = now_mask;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    } catch (const std::exception& error) {
      runtime_session.notice("ERROR", error.what());
      runtime_session.result("exception");
      throw;
    }
  } catch (const std::exception& e) {
    std::cerr << "[action_go_remote_test] ERROR: " << e.what() << std::endl;
    return 1;
  }
}
