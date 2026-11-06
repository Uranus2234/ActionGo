#include "action_go/runtime/activation_blend.h"
#include "action_go/runtime/command_safety_filter.h"
#include "action_go/runtime/event_logger.h"
#include "action_go/runtime/command_timing.h"
#include "action_go/runtime/npy_reader.h"
#include "action_go/runtime/policy_protocol.h"
#include "action_go/runtime/preview_lifecycle.h"
#include "action_go/runtime/yaml_contract.h"

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>
#include <zlib.h>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

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
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

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
using action_go::runtime::kOmniXtremeProtocolCommandObsDim;
using action_go::runtime::kOmniXtremeProtocolHistoryDim;
using action_go::runtime::kOmniXtremeProtocolRealObsDim;
using action_go::runtime::kOmniXtremeProtocolResidualObsDim;
using action_go::runtime::kUnitreeG1ProtocolMotorCount;
namespace yamlc = action_go::runtime::yaml_contract;

constexpr int kNumJoints = kUnitreeG1ProtocolMotorCount;
constexpr int kRealObsDim = kOmniXtremeProtocolRealObsDim;
constexpr int kCommandObsDim = kOmniXtremeProtocolCommandObsDim;
constexpr int kHistDim = kOmniXtremeProtocolHistoryDim;
constexpr int kHistSteps = kHistDim / kRealObsDim;
constexpr int kResidualObsDim = kOmniXtremeProtocolResidualObsDim;

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
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

std::vector<uint8_t> read_binary_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) throw std::runtime_error("failed to open file: " + path.string());
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  if (size < 0) throw std::runtime_error("failed to stat file: " + path.string());
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!data.empty()) in.read(reinterpret_cast<char*>(data.data()), size);
  if (!in.good() && !in.eof()) throw std::runtime_error("failed to read file: " + path.string());
  return data;
}

bool looks_like_git_lfs_pointer(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) return false;
  char buf[256]{};
  in.read(buf, sizeof(buf) - 1);
  const std::string head(buf, static_cast<size_t>(std::max<std::streamsize>(0, in.gcount())));
  return head.find("version https://git-lfs.github.com/spec/v1") != std::string::npos;
}

void require_real_asset_file(const fs::path& path, const std::string& what) {
  if (!fs::exists(path)) {
    throw std::runtime_error(what + " file does not exist: " + path.string());
  }
  if (looks_like_git_lfs_pointer(path)) {
    throw std::runtime_error(
        what + " is a Git LFS pointer, not the real asset: " + path.string() +
        ". Run `git lfs pull` on the robot, or copy the real file with rsync/scp.");
  }
  const auto size = fs::file_size(path);
  if (size == 0) {
    throw std::runtime_error(what + " file is empty: " + path.string());
  }
}

uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

std::vector<uint8_t> inflate_raw_deflate(const uint8_t* src, size_t src_size, size_t dst_size) {
  std::vector<uint8_t> dst(dst_size);
  z_stream zs{};
  zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(src));
  zs.avail_in = static_cast<uInt>(src_size);
  zs.next_out = reinterpret_cast<Bytef*>(dst.data());
  zs.avail_out = static_cast<uInt>(dst.size());
  int ret = inflateInit2(&zs, -MAX_WBITS);
  if (ret != Z_OK) throw std::runtime_error("inflateInit2 failed");
  ret = inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  if (ret != Z_STREAM_END || zs.total_out != dst_size) throw std::runtime_error("zip deflate inflate failed");
  return dst;
}

std::unordered_map<std::string, std::vector<uint8_t>> read_npz_entries(const fs::path& path) {
  const auto bytes = read_binary_file(path);
  std::unordered_map<std::string, std::vector<uint8_t>> entries;
  size_t pos = 0;
  while (pos + 30 <= bytes.size()) {
    const uint32_t sig = le32(bytes.data() + pos);
    if (sig == 0x02014b50 || sig == 0x06054b50) break;
    if (sig != 0x04034b50) throw std::runtime_error("invalid zip local header in " + path.string());
    const uint16_t flags = le16(bytes.data() + pos + 6);
    const uint16_t method = le16(bytes.data() + pos + 8);
    const uint32_t comp_size = le32(bytes.data() + pos + 18);
    const uint32_t uncomp_size = le32(bytes.data() + pos + 22);
    const uint16_t name_len = le16(bytes.data() + pos + 26);
    const uint16_t extra_len = le16(bytes.data() + pos + 28);
    if (flags & 0x08) throw std::runtime_error("npz uses data descriptors; unsupported: " + path.string());
    const size_t name_pos = pos + 30;
    const size_t data_pos = name_pos + name_len + extra_len;
    if (data_pos + comp_size > bytes.size()) throw std::runtime_error("truncated zip entry");
    std::string name(reinterpret_cast<const char*>(bytes.data() + name_pos), name_len);
    std::vector<uint8_t> payload;
    if (method == 0) payload.assign(bytes.begin() + data_pos, bytes.begin() + data_pos + comp_size);
    else if (method == 8) payload = inflate_raw_deflate(bytes.data() + data_pos, comp_size, uncomp_size);
    else throw std::runtime_error("unsupported npz compression method " + std::to_string(method));
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".npy") name = name.substr(0, name.size() - 4);
    entries[name] = std::move(payload);
    pos = data_pos + comp_size;
  }
  if (entries.empty()) throw std::runtime_error("no npy entries in npz: " + path.string());
  return entries;
}

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string lower_string(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool contains_str(const std::vector<std::string>& xs, const std::string& x) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}

std::string join_strings(const std::vector<std::string>& xs, const std::string& sep = ",") {
  std::ostringstream os;
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i) os << sep;
    os << xs[i];
  }
  return os.str();
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ",";
    os << shape[i];
  }
  os << "]";
  return os.str();
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

action_go::runtime::npy::ArrayView npy_meta(
    const std::unordered_map<std::string, std::vector<uint8_t>>& npz,
    const std::string& key) {
  return action_go::runtime::npy::metadata(npz, key);
}

std::vector<float> npy_to_float_vector(const std::vector<uint8_t>& bytes, const std::string& name) {
  return action_go::runtime::npy::to_float_vector(bytes, name, true);
}

std::vector<float> yaml_float_vec_or(const YAML::Node& root, const std::string& key, const std::vector<float>& fallback) {
  if (!root || !root[key]) return fallback;
  std::vector<float> out;
  for (const auto& v : root[key]) out.push_back(v.as<float>());
  return out;
}

std::array<float, 4> normalize_quat(std::array<float, 4> q) {
  const float n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  if (!(n > 1e-8f) || !std::isfinite(n)) return {1.f, 0.f, 0.f, 0.f};
  for (float& v : q) v /= n;
  return q;
}

std::array<float, 4> quat_mul(const std::array<float, 4>& a_in, const std::array<float, 4>& b_in) {
  const auto a = normalize_quat(a_in);
  const auto b = normalize_quat(b_in);
  const float aw=a[0], ax=a[1], ay=a[2], az=a[3];
  const float bw=b[0], bx=b[1], by=b[2], bz=b[3];
  return normalize_quat({
      aw*bw - ax*bx - ay*by - az*bz,
      aw*bx + ax*bw + ay*bz - az*by,
      aw*by - ax*bz + ay*bw + az*bx,
      aw*bz + ax*by - ay*bx + az*bw});
}

std::array<float, 4> quat_conj(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  return {q[0], -q[1], -q[2], -q[3]};
}

float quat_yaw(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w=q[0], x=q[1], y=q[2], z=q[3];
  return std::atan2(2.f * (w*z + x*y), 1.f - 2.f * (y*y + z*z));
}

std::array<float, 3> quat_rpy(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w=q[0], x=q[1], y=q[2], z=q[3];
  const float roll = std::atan2(2.f * (w*x + y*z), 1.f - 2.f * (x*x + y*y));
  const float sinp = 2.f * (w*y - z*x);
  const float pitch = std::asin(std::clamp(sinp, -1.f, 1.f));
  const float yaw = std::atan2(2.f * (w*z + x*y), 1.f - 2.f * (y*y + z*z));
  return {roll, pitch, yaw};
}

std::array<float, 4> quat_from_rpy(float roll, float pitch, float yaw) {
  const float cr = std::cos(0.5f * roll), sr = std::sin(0.5f * roll);
  const float cp = std::cos(0.5f * pitch), sp = std::sin(0.5f * pitch);
  const float cy = std::cos(0.5f * yaw), sy = std::sin(0.5f * yaw);
  return normalize_quat({
      cr*cp*cy + sr*sp*sy,
      sr*cp*cy - cr*sp*sy,
      cr*sp*cy + sr*cp*sy,
      cr*cp*sy - sr*sp*cy});
}

std::array<float, 9> matrix_from_quat(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w=q[0], x=q[1], y=q[2], z=q[3];
  return {{
      1.f - 2.f * (y*y + z*z), 2.f * (x*y - z*w),       2.f * (x*z + y*w),
      2.f * (x*y + z*w),       1.f - 2.f * (x*x + z*z), 2.f * (y*z - x*w),
      2.f * (x*z - y*w),       2.f * (y*z + x*w),       1.f - 2.f * (x*x + y*y)}};
}

std::vector<float> to_vec(const std::array<float, kNumJoints>& a) { return std::vector<float>(a.begin(), a.end()); }

void check_vec_size(const std::string& name, size_t n, const std::vector<float>& v) {
  if (v.size() != n) throw std::runtime_error(name + " size mismatch: got " + std::to_string(v.size()) + " expected " + std::to_string(n));
}

std::vector<float> lerp_vec(const std::vector<float>& a, const std::vector<float>& b, double alpha) {
  if (a.size() != b.size()) return b;
  alpha = std::clamp(alpha, 0.0, 1.0);
  std::vector<float> out(a.size(), 0.f);
  for (size_t i = 0; i < a.size(); ++i) out[i] = static_cast<float>((1.0 - alpha) * a[i] + alpha * b[i]);
  return out;
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

struct Args {
  std::string net{"enP8p1s0"};
  int domain_id{0};
  std::string base_model;
  std::string residual_model;
  std::string fk_model;
  std::string params;
  std::string motion;
  std::string cmd_topic{"rt/user_lowcmd"};
  std::string state_topic{"rt/lowstate"};
  std::string ready_file;
  std::string start_file;
  std::string preview_file;
  std::string commit_file;
  std::string discard_file;
  std::string event_log;
  std::string execution_provider{"cpu"}; // cpu / cuda / tensorrt / auto
  std::string trt_cache_dir{"/tmp/action_go_omnixtreme_trt_cache"};
  int inference_hz{50};
  int publish_hz{500};
  int onnx_threads{2};
  int gpu_device_id{0};
  int mode_machine{5};
  int motion_start_frame{0};
  int max_frames{0};
  int stats_interval{50};
  int warmup_iters{1};
  double duration_sec{0.0};
  double prepare_ramp_sec{0.0};
  double prepare_hold_sec{0.0};
  double startup_interp_sec{0.0};
  double target_interp_sec{0.0};
  double activation_blend_sec{0.10};
  double preview_timeout_sec{5.0};
  double residual_gain{1.0};
  double initial_noise_scale{1.0};
  size_t trt_workspace_size{1ull << 30}; // 1 GiB
  bool deterministic{false};
  bool trt_fp16{true};
  bool clip_envelope{true};
  bool joint_limit_clamp{false};
  bool safety_joint_limit_enabled{true};
  bool friction_feedforward{true};
  bool sync_lowstate{false};
  bool check_only{false};
  bool dry_run{false};
  bool no_damping_on_stop{true};
  bool experimental_no_prestart_hold{false};
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (k == "--net") a.net = need("--net");
    else if (k == "--domain-id") a.domain_id = std::stoi(need("--domain-id"));
    else if (k == "--model" || k == "--base-model") a.base_model = need(k.c_str());
    else if (k == "--residual-model") a.residual_model = need("--residual-model");
    else if (k == "--fk-model") a.fk_model = need("--fk-model");
    else if (k == "--params" || k == "--yaml") a.params = need(k.c_str());
    else if (k == "--trajectory" || k == "--motion") a.motion = need(k.c_str());
    else if (k == "--cmd-topic") a.cmd_topic = need("--cmd-topic");
    else if (k == "--state-topic") a.state_topic = need("--state-topic");
    else if (k == "--ready-file") a.ready_file = need("--ready-file");
    else if (k == "--start-file") a.start_file = need("--start-file");
    else if (k == "--preview-file") a.preview_file = need("--preview-file");
    else if (k == "--commit-file") a.commit_file = need("--commit-file");
    else if (k == "--discard-file") a.discard_file = need("--discard-file");
    else if (k == "--event-log") a.event_log = need("--event-log");
    else if (k == "--ep" || k == "--execution-provider") a.execution_provider = need(k.c_str());
    else if (k == "--gpu-device-id" || k == "--device-id") a.gpu_device_id = std::stoi(need(k.c_str()));
    else if (k == "--trt-cache-dir" || k == "--trt-engine-cache-dir") a.trt_cache_dir = need(k.c_str());
    else if (k == "--trt-workspace-size") a.trt_workspace_size = static_cast<size_t>(std::stoull(need(k.c_str())));
    else if (k == "--trt-fp16") a.trt_fp16 = true;
    else if (k == "--no-trt-fp16" || k == "--trt-fp32") a.trt_fp16 = false;
    else if (k == "--hz") a.inference_hz = std::stoi(need("--hz"));
    else if (k == "--publish-hz") a.publish_hz = std::stoi(need("--publish-hz"));
    else if (k == "--onnx-threads") a.onnx_threads = std::stoi(need("--onnx-threads"));
    else if (k == "--mode-machine") a.mode_machine = std::stoi(need("--mode-machine"));
    else if (k == "--motion-start-frame" || k == "--motion-start-step") a.motion_start_frame = std::stoi(need(k.c_str()));
    else if (k == "--max-frames" || k == "--max-steps" || k == "--motion-end-step" || k == "--motion-end-frame") a.max_frames = std::stoi(need(k.c_str()));
    else if (k == "--duration-sec") a.duration_sec = std::stod(need("--duration-sec"));
    else if (k == "--startup-interp-sec") a.startup_interp_sec = std::stod(need("--startup-interp-sec"));
    else if (k == "--target-interp-sec") a.target_interp_sec = std::stod(need("--target-interp-sec"));
    else if (k == "--activation-blend-sec") a.activation_blend_sec = std::stod(need("--activation-blend-sec"));
    else if (k == "--preview-timeout-sec") a.preview_timeout_sec = std::stod(need("--preview-timeout-sec"));
    else if (k == "--residual-gain") a.residual_gain = std::stod(need("--residual-gain"));
    else if (k == "--initial-noise-scale") a.initial_noise_scale = std::stod(need("--initial-noise-scale"));
    else if (k == "--stats-interval") a.stats_interval = std::stoi(need("--stats-interval"));
    else if (k == "--warmup-iters") a.warmup_iters = std::stoi(need("--warmup-iters"));
    else if (k == "--no-warmup") a.warmup_iters = 0;
    else if (k == "--prepare-ramp-sec") a.prepare_ramp_sec = std::stod(need("--prepare-ramp-sec"));
    else if (k == "--prepare-hold-sec") a.prepare_hold_sec = std::stod(need("--prepare-hold-sec"));
    else if (k == "--no-prepare") { a.prepare_ramp_sec = 0.0; a.prepare_hold_sec = 0.0; }
    else if (k == "--deterministic" || k == "--zero-initial-noise") a.deterministic = true;
    else if (k == "--clip-envelope") a.clip_envelope = true;
    else if (k == "--no-clip-envelope") a.clip_envelope = false;
    else if (k == "--joint-limit-clamp") a.joint_limit_clamp = true;
    else if (k == "--no-joint-limit-clamp") a.joint_limit_clamp = false;
    else if (k == "--safety-joint-limit") a.safety_joint_limit_enabled = true;
    else if (k == "--no-safety-joint-limit") a.safety_joint_limit_enabled = false;
    else if (k == "--friction-feedforward") a.friction_feedforward = true;
    else if (k == "--no-friction-feedforward") a.friction_feedforward = false;
    else if (k == "--sync-lowstate") a.sync_lowstate = true;
    else if (k == "--check-only") a.check_only = true;
    else if (k == "--dry-run" || k == "--dds-dry-run") a.dry_run = true;
    else if (k == "--no-damping-on-stop") a.no_damping_on_stop = true;
    else if (k == "--damping-on-stop") a.no_damping_on_stop = false;
    else if (k == "--experimental-no-prestart-hold") a.experimental_no_prestart_hold = true;
    else if (k == "--help") {
      std::cout << "Usage: generic_composite_policy_player --spec actions/tracking/omnixtreme/config/generic_omnixtreme.yaml [runtime overrides]\n"
                << "  C++ ONNXRuntime/Unitree DDS player for OmniXtreme G1 29-DOF policy.\n"
                << "  --cmd-topic rt/user_lowcmd      G1 UserCtrl command topic; sim uses rt/lowcmd\n"
                << "  --state-topic rt/lowstate       LowState topic\n"
                << "  --ready-file PATH --start-file PATH  action_go prestart handoff\n"
                << "  --ep cpu|cuda|tensorrt|auto       ONNX Runtime execution provider (default cpu)\n"
                << "  --trt-fp16 --trt-cache-dir DIR     TensorRT options; cache avoids rebuilding engines\n"
                << "  --hz 50 --publish-hz 500 --onnx-threads 2\n"
                << "  --warmup-iters N                run N inference warmups before ready/start (default 1)\n"
                << "  --prepare-ramp-sec S --prepare-hold-sec S  ramp to motion start pose after UserCtrl, before policy\n"
                << "  --max-frames N                  stop after N policy frames; 0 means full motion\n"
                << "  --deterministic                 feed zero initial_noise to base policy\n"
                << "  --friction-feedforward          publish deploy_mujoco.py friction compensation tau (default)\n"
                << "  --joint-limit-clamp             clamp q targets to YAML joint limits (off by default; deploy_mujoco.py does not do this)\n"
                << "  --no-safety-joint-limit         disable shared LowCmd mechanical q-target clamp\n"
                << "  --no-friction-feedforward       disable feedforward tau\n"
                << "  --sync-lowstate                 simulation mode: one publish tick per received LowState\n"
                << "  --dry-run                       run DDS/ONNX but do not publish LowCmd\n"
                << "  --experimental-no-prestart-hold unsafe handoff-ablation baseline\n"
                << "  --check-only                    load model/motion and run one synthetic inference without DDS\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }
  if (a.base_model.empty() || a.residual_model.empty() || a.fk_model.empty() || a.motion.empty()) {
    throw std::runtime_error("--model, --residual-model, --fk-model and --trajectory/--motion are required");
  }
  if (!a.preview_file.empty() || !a.commit_file.empty() || !a.discard_file.empty()) {
    action_go::runtime::validate_preview_lifecycle(
        a.preview_file, a.commit_file, a.discard_file,
        a.preview_timeout_sec, a.cmd_topic);
  }
  return a;
}

} // namespace

class MotionCache {
public:
  explicit MotionCache(const std::string& path) {
    auto entries = read_npz_entries(path);
    auto meta_jp = npy_meta(entries, "joint_pos");
    auto meta_jv = npy_meta(entries, "joint_vel");
    auto meta_bq = npy_meta(entries, "body_quat_w");
    if (meta_jp.shape.size() != 2 || meta_jp.shape[1] != kNumJoints) {
      throw std::runtime_error("OmniXtreme motion joint_pos shape must be [frames,29]");
    }
    if (meta_jv.shape.size() != 2 || meta_jv.shape[1] != kNumJoints) {
      throw std::runtime_error("OmniXtreme motion joint_vel shape must be [frames,29]");
    }
    if (meta_bq.shape.size() != 3 || meta_bq.shape[2] != 4) {
      throw std::runtime_error("OmniXtreme motion body_quat_w shape must be [frames,bodies,4]");
    }
    num_frames = static_cast<int>(meta_jp.shape[0]);
    num_bodies = static_cast<int>(meta_bq.shape[1]);
    joint_pos_raw = npy_to_float_vector(entries.at("joint_pos"), "joint_pos");
    joint_vel_raw = npy_to_float_vector(entries.at("joint_vel"), "joint_vel");
    body_quat_w = npy_to_float_vector(entries.at("body_quat_w"), "body_quat_w");
    if (entries.count("fps")) fps = npy_to_float_vector(entries.at("fps"), "fps").at(0);
    if (!std::isfinite(fps) || fps <= 0.f) throw std::runtime_error("OmniXtreme motion fps must be positive and finite");
    if (static_cast<int>(joint_pos_raw.size()) != num_frames * kNumJoints ||
        static_cast<int>(joint_vel_raw.size()) != num_frames * kNumJoints ||
        static_cast<int>(body_quat_w.size()) != num_frames * num_bodies * 4) {
      throw std::runtime_error("OmniXtreme motion array size mismatch");
    }
  }

  void set_joint_permutation(std::vector<int> permutation) {
    yamlc::require_permutation("robot.joint_permutation", permutation, kNumJoints);
    joint_permutation_ = std::move(permutation);
  }

  void dof_at(int frame, std::vector<float>& pos, std::vector<float>& vel) const {
    yamlc::require_size("robot.joint_permutation", kNumJoints, joint_permutation_.size());
    const int f = std::clamp(frame, 0, num_frames - 1);
    pos.assign(kNumJoints, 0.f);
    vel.assign(kNumJoints, 0.f);
    const size_t base = static_cast<size_t>(f) * kNumJoints;
    for (int i = 0; i < kNumJoints; ++i) {
      const int src = joint_permutation_[static_cast<size_t>(i)];
      pos[i] = joint_pos_raw[base + static_cast<size_t>(src)];
      vel[i] = joint_vel_raw[base + static_cast<size_t>(src)];
    }
  }

  std::array<float,4> body_quat_at(int frame, int body) const {
    const int f = std::clamp(frame, 0, num_frames - 1);
    const int b = std::clamp(body, 0, num_bodies - 1);
    const size_t off = (static_cast<size_t>(f) * num_bodies + b) * 4;
    return normalize_quat({body_quat_w[off], body_quat_w[off+1], body_quat_w[off+2], body_quat_w[off+3]});
  }

  int num_frames{0};
  int num_bodies{0};
  float fps{50.f};
  std::vector<float> joint_pos_raw, joint_vel_raw, body_quat_w;
  std::vector<int> joint_permutation_;
};

class OmniXtremePolicy {
public:
  struct StateSnapshot {
    int frame{0};
    std::vector<float> history;
    std::vector<float> prev_action;
    std::vector<float> prev_base_action;
    std::mt19937 rng;
    bool heading_initialized{false};
    float yaw_offset{0.f};
  };

  struct RobotState {
    std::array<float,4> quat{1.f,0.f,0.f,0.f}; // Unitree wxyz
    std::array<float,3> gyro{0.f,0.f,0.f};
    std::vector<float> q, dq;
    int mode_machine{0};
    std::chrono::steady_clock::time_point recv_stamp{};
    uint64_t seq{0};
  };

  explicit OmniXtremePolicy(const Args& args)
      : args_(args), motion_(args.motion), env_(ORT_LOGGING_LEVEL_WARNING, "generic_composite_policy_player") {
    if (std::abs(motion_.fps - static_cast<float>(args_.inference_hz)) > 0.5f) {
      throw std::runtime_error("OmniXtreme motion fps must match --hz within 0.5 Hz");
    }
    load_params(args.params);
    load_sessions();
    history_.assign(kHistDim, 0.f);
    prev_action_.assign(kNumJoints, 0.f);
    prev_base_action_.assign(kNumJoints, 0.f);
    reset_runtime_state();
  }

  void check_only() {
    RobotState s;
    motion_.dof_at(0, s.q, s.dq);
    s.dq.assign(kNumJoints, 0.f);
    s.quat = motion_.body_quat_at(0, 0);
    s.gyro = {0.f, 0.f, 0.f};
    const auto q = infer(s);
    const auto [mn, mx] = std::minmax_element(q.begin(), q.end());
    std::cout << "[generic_composite_policy_player][omnixtreme] check-only ok\n"
              << "  motion_frames=" << motion_.num_frames << " fps=" << motion_.fps
              << " bodies=" << motion_.num_bodies << " anchor_body=" << anchor_body_index_ << "\n"
              << "  base=" << args_.base_model << "\n"
              << "  residual=" << args_.residual_model << " gain=" << residual_gain_ << "\n"
              << "  fk=" << args_.fk_model << "\n"
              << "  first_target_range=[" << *mn << "," << *mx << "] frame=" << frame_ << "\n";
  }

  void warmup(const RobotState& s, int iters) {
    if (iters <= 0) return;
    std::cout << "[generic_composite_policy_player][omnixtreme] warmup before ready/start: iters=" << iters << "\n";
    double sum_ms = 0.0;
    double max_ms = 0.0;
    double last_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      (void)infer(s);
      const auto t1 = std::chrono::steady_clock::now();
      last_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      sum_ms += last_ms;
      max_ms = std::max(max_ms, last_ms);
    }
    reset_runtime_state();
    std::cout << "[generic_composite_policy_player][omnixtreme] warmup done ms(last/avg/max)="
              << last_ms << "/" << (sum_ms / static_cast<double>(iters)) << "/" << max_ms
              << " runtime state reset\n";
  }

  std::vector<float> infer(const RobotState& s) {
    if (s.q.size() < kNumJoints || s.dq.size() < kNumJoints) {
      throw std::runtime_error("lowstate motor count too small");
    }
    if (!heading_initialized_) init_heading(s.quat);

    const int ref_frame = std::clamp(frame_ + 1, 0, motion_.num_frames - 1);
    std::vector<float> ref_q, ref_dq;
    motion_.dof_at(ref_frame, ref_q, ref_dq);

    std::vector<float> q(s.q.begin(), s.q.begin() + kNumJoints);
    std::vector<float> dq(s.dq.begin(), s.dq.begin() + kNumJoints);
    std::vector<float> dof_rel(kNumJoints, 0.f);
    for (int i = 0; i < kNumJoints; ++i) dof_rel[i] = q[i] - default_pos_[i];
    std::array<float, 3> gyro = s.gyro;
    if (!args_.deterministic && base_ang_vel_noise_ > 0.f) {
      std::uniform_real_distribution<float> ud(-base_ang_vel_noise_, base_ang_vel_noise_);
      for (float& v : gyro) v = (v + ud(rng_)) * base_ang_vel_scale_;
    } else {
      for (float& v : gyro) v *= base_ang_vel_scale_;
    }

    std::vector<float> real;
    real.reserve(kRealObsDim);
    real.insert(real.end(), dof_rel.begin(), dof_rel.end());
    real.insert(real.end(), dq.begin(), dq.end());
    real.insert(real.end(), gyro.begin(), gyro.end());
    real.insert(real.end(), prev_base_action_.begin(), prev_base_action_.end());
    if (real.size() != kRealObsDim) throw std::runtime_error("internal real_obs dim mismatch");

    std::copy(history_.begin() + kRealObsDim, history_.end(), history_.begin());
    std::copy(real.begin(), real.end(), history_.end() - kRealObsDim);

    const auto corrected_base = corrected_base_quat(s.quat);
    const auto current_anchor = run_fk_anchor(q, corrected_base);
    const auto ref_anchor = motion_.body_quat_at(ref_frame, anchor_body_index_);
    const auto rel_anchor = quat_mul(quat_conj(current_anchor), ref_anchor);
    const auto R = matrix_from_quat(rel_anchor);
    std::array<float, 6> anchor6{};
    int aidx = 0;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 2; ++c) anchor6[aidx++] = R[r * 3 + c];
    }

    std::vector<float> command;
    command.reserve(kCommandObsDim);
    command.insert(command.end(), ref_q.begin(), ref_q.end());
    command.insert(command.end(), ref_dq.begin(), ref_dq.end());
    command.insert(command.end(), anchor6.begin(), anchor6.end());
    if (command.size() != kCommandObsDim) throw std::runtime_error("internal command_obs dim mismatch");

    std::vector<float> noise(kNumJoints, 0.f);
    if (!args_.deterministic && initial_noise_scale_ > 0.0) {
      std::normal_distribution<float> nd(0.f, static_cast<float>(initial_noise_scale_));
      for (float& v : noise) v = nd(rng_);
    }
    const auto base_action = run_base(real, command, history_, noise);

    std::vector<float> residual_obs;
    residual_obs.reserve(kResidualObsDim);
    for (int i = 0; i < kNumJoints; ++i) residual_obs.push_back(ref_q[inverse_joint_permutation_[static_cast<size_t>(i)]]);
    for (int i = 0; i < kNumJoints; ++i) residual_obs.push_back(ref_dq[inverse_joint_permutation_[static_cast<size_t>(i)]]);
    residual_obs.insert(residual_obs.end(), anchor6.begin(), anchor6.end());
    residual_obs.insert(residual_obs.end(), gyro.begin(), gyro.end());
    for (int i = 0; i < kNumJoints; ++i) residual_obs.push_back(dof_rel[inverse_joint_permutation_[static_cast<size_t>(i)]]);
    for (int i = 0; i < kNumJoints; ++i) residual_obs.push_back(dq[inverse_joint_permutation_[static_cast<size_t>(i)]]);
    for (int i = 0; i < kNumJoints; ++i) residual_obs.push_back(prev_action_[inverse_joint_permutation_[static_cast<size_t>(i)]]);
    for (int i = 0; i < kNumJoints; ++i) residual_obs.push_back(base_action[inverse_joint_permutation_[static_cast<size_t>(i)]]);
    if (residual_obs.size() != kResidualObsDim) throw std::runtime_error("internal residual_obs dim mismatch");

    const auto residual_proto = run_residual(residual_obs);
    std::vector<float> raw_action(kNumJoints, 0.f);
    for (int i = 0; i < kNumJoints; ++i) {
      const float residual_env = residual_proto[joint_permutation_[static_cast<size_t>(i)]];
      raw_action[i] = base_action[i] + static_cast<float>(residual_gain_) * residual_env;
      if (!std::isfinite(raw_action[i])) throw std::runtime_error("policy produced non-finite action");
    }

    std::vector<float> q_target(kNumJoints, 0.f);
    for (int i = 0; i < kNumJoints; ++i) q_target[i] = raw_action[i] * action_scale_[i] + default_pos_[i];
    if (args_.clip_envelope) clip_with_envelope(q, dq, q_target);
    if (args_.joint_limit_clamp) {
      for (int i = 0; i < kNumJoints; ++i) q_target[i] = std::clamp(q_target[i], lower_limits_[i], upper_limits_[i]);
    }

    prev_action_ = std::move(raw_action);
    prev_base_action_ = base_action;
    return q_target;
  }

  void advance(bool loop=false) {
    if (frame_ < motion_.num_frames - 2) ++frame_;
    else if (loop) frame_ = args_.motion_start_frame;
  }
  bool done() const { return frame_ >= motion_.num_frames - 2; }
  int frame() const { return frame_; }
  int total_frames() const { return motion_.num_frames; }
  const std::vector<float>& kp() const { return kp_; }
  const std::vector<float>& kd() const { return kd_; }
  const std::vector<float>& lower_limits() const { return lower_limits_; }
  const std::vector<float>& upper_limits() const { return upper_limits_; }
  const std::vector<float>& friction_va() const { return friction_va_; }
  const std::vector<float>& friction_fs() const { return friction_fs_; }
  const std::vector<float>& friction_fd() const { return friction_fd_; }
  const std::vector<float>& torque_limits() const { return torque_limits_; }
  std::vector<float> motion_start_q() const {
    std::vector<float> q, dq;
    motion_.dof_at(frame_, q, dq);
    return q;
  }
  StateSnapshot snapshot_state() const {
    return {frame_, history_, prev_action_, prev_base_action_, rng_,
            heading_initialized_, yaw_offset_};
  }
  void restore_state(const StateSnapshot& snapshot) {
    frame_ = snapshot.frame;
    history_ = snapshot.history;
    prev_action_ = snapshot.prev_action;
    prev_base_action_ = snapshot.prev_base_action;
    rng_ = snapshot.rng;
    heading_initialized_ = snapshot.heading_initialized;
    yaw_offset_ = snapshot.yaw_offset;
  }

private:
  void reset_runtime_state() {
    frame_ = std::clamp(args_.motion_start_frame, 0, std::max(0, motion_.num_frames - 2));
    std::fill(history_.begin(), history_.end(), 0.f);
    std::fill(prev_action_.begin(), prev_action_.end(), 0.f);
    std::fill(prev_base_action_.begin(), prev_base_action_.end(), 0.f);
    rng_.seed(12345u);
    heading_initialized_ = false;
    yaw_offset_ = 0.f;
  }

  void load_params(const std::string& path) {
    if (path.empty() || !fs::exists(path)) {
      throw std::runtime_error("OmniXtreme params not found: " + path);
    }
    const YAML::Node y = YAML::LoadFile(path);
    yamlc::require_size("robot.motor_count", kNumJoints,
                        static_cast<size_t>(yamlc::required_scalar<int>(y, "robot.motor_count")));
    joint_permutation_ = yamlc::required_vector_size<int>(y, "robot.joint_permutation", kNumJoints);
    inverse_joint_permutation_ = yamlc::required_vector_size<int>(y, "robot.inverse_joint_permutation", kNumJoints);
    yamlc::require_permutation("robot.joint_permutation", joint_permutation_, kNumJoints);
    yamlc::require_permutation("robot.inverse_joint_permutation", inverse_joint_permutation_, kNumJoints);
    motion_.set_joint_permutation(joint_permutation_);
    default_pos_ = yamlc::required_vector_size<float>(y, "robot.default_angles", kNumJoints);
    action_scale_ = yamlc::required_vector_size<float>(y, "robot.action_scale", kNumJoints);
    kp_ = yamlc::required_vector_size<float>(y, "robot.kp", kNumJoints);
    kd_ = yamlc::required_vector_size<float>(y, "robot.kd", kNumJoints);
    torque_limits_ = yamlc::required_vector_size<float>(y, "robot.torque_limits", kNumJoints);
    lower_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_lower_limit", kNumJoints);
    upper_limits_ = yamlc::required_vector_size<float>(y, "robot.joint_pos_upper_limit", kNumJoints);
    friction_x1_ = yamlc::required_vector_size<float>(y, "safety.friction.x1", kNumJoints);
    friction_x2_ = yamlc::required_vector_size<float>(y, "safety.friction.x2", kNumJoints);
    friction_y1_ = yamlc::required_vector_size<float>(y, "safety.friction.y1", kNumJoints);
    friction_y2_ = yamlc::required_vector_size<float>(y, "safety.friction.y2", kNumJoints);
    friction_va_ = yamlc::required_vector_size<float>(y, "safety.friction.va", kNumJoints);
    friction_fs_ = yamlc::required_vector_size<float>(y, "safety.friction.fs", kNumJoints);
    friction_fd_ = yamlc::required_vector_size<float>(y, "safety.friction.fd", kNumJoints);
    anchor_body_index_ = yamlc::required_scalar<int>(y, "motion.anchor_body_index");
    residual_gain_ = args_.residual_gain;
    if (yamlc::optional_node(y, "action.residual_gain")) {
      residual_gain_ = yamlc::required_scalar<double>(y, "action.residual_gain");
    }
    initial_noise_scale_ = args_.initial_noise_scale;
    if (args_.initial_noise_scale == 1.0) {
      initial_noise_scale_ = yamlc::required_scalar<double>(y, "action.initial_noise_scale");
    }
    base_ang_vel_noise_ = yamlc::required_scalar<float>(y, "observation.base_ang_vel_noise");
    base_ang_vel_scale_ = yamlc::required_scalar<float>(y, "observation.base_ang_vel_scale");
  }

  void load_sessions() {
    auto opt_base = make_session_options("base");
    base_session_ = Ort::Session(env_, args_.base_model.c_str(), opt_base);
    auto opt_residual = make_session_options("residual");
    residual_session_ = Ort::Session(env_, args_.residual_model.c_str(), opt_residual);
    auto opt_fk = make_session_options("fk");
    fk_session_ = Ort::Session(env_, args_.fk_model.c_str(), opt_fk);
  }

  Ort::SessionOptions make_session_options(const std::string& session_name) const {
    Ort::SessionOptions opt;
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opt.SetIntraOpNumThreads(std::max(1, args_.onnx_threads));
    opt.SetInterOpNumThreads(1);

    const auto providers = Ort::GetAvailableProviders();
    const std::string requested = lower_string(trim(args_.execution_provider));
    const bool has_trt = contains_str(providers, "TensorrtExecutionProvider");
    const bool has_cuda = contains_str(providers, "CUDAExecutionProvider");
    std::cout << "[generic_composite_policy_player][omnixtreme] ORT available providers: "
              << join_strings(providers, ",") << "\n";

    auto append_cuda = [&]() -> bool {
      try {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = args_.gpu_device_id;
        opt.AppendExecutionProvider_CUDA(cuda);
        std::cout << "[generic_composite_policy_player][omnixtreme] " << session_name
                  << " appended CUDAExecutionProvider device=" << args_.gpu_device_id << "\n";
        return true;
      } catch (const Ort::Exception& e) {
        std::cerr << "[generic_composite_policy_player][omnixtreme] " << session_name
                  << " CUDA EP append failed: " << e.what() << "\n";
        return false;
      }
    };

    auto append_trt = [&]() -> bool {
      try {
        if (!args_.trt_cache_dir.empty()) fs::create_directories(args_.trt_cache_dir);
        OrtTensorRTProviderOptions trt{};
        trt.device_id = args_.gpu_device_id;
        trt.trt_max_partition_iterations = 1000;
        trt.trt_min_subgraph_size = 1;
        trt.trt_max_workspace_size = args_.trt_workspace_size;
        trt.trt_fp16_enable = args_.trt_fp16 ? 1 : 0;
        trt.trt_engine_cache_enable = args_.trt_cache_dir.empty() ? 0 : 1;
        trt.trt_engine_cache_path = args_.trt_cache_dir.empty() ? nullptr : args_.trt_cache_dir.c_str();
        trt.trt_force_sequential_engine_build = 1;
        opt.AppendExecutionProvider_TensorRT(trt);
        std::cout << "[generic_composite_policy_player][omnixtreme] " << session_name
                  << " appended TensorrtExecutionProvider device=" << args_.gpu_device_id
                  << " fp16=" << (args_.trt_fp16 ? "true" : "false")
                  << " cache=" << (args_.trt_cache_dir.empty() ? "(off)" : args_.trt_cache_dir) << "\n";
        return true;
      } catch (const Ort::Exception& e) {
        std::cerr << "[generic_composite_policy_player][omnixtreme] " << session_name
                  << " TensorRT EP append failed: " << e.what() << "\n";
        return false;
      }
    };

    if (requested == "cpu" || requested.empty()) {
      std::cout << "[generic_composite_policy_player][omnixtreme] " << session_name << " using CPUExecutionProvider\n";
      return opt;
    }
    if (requested == "cuda") {
      if (!has_cuda) {
        throw std::runtime_error(
            "--ep cuda requested, but ONNX Runtime does not list CUDAExecutionProvider. "
            "Install/link an onnxruntime-gpu build for Jetson.");
      }
      if (!append_cuda()) throw std::runtime_error("--ep cuda requested but CUDA EP append failed");
      return opt;
    }
    if (requested == "tensorrt" || requested == "trt") {
      if (!has_trt) {
        throw std::runtime_error(
            "--ep tensorrt requested, but ONNX Runtime does not list TensorrtExecutionProvider. "
            "The bundled thirdparty ONNX Runtime is CPU-only; replace ACTION_GO_ONNXRUNTIME_DIR "
            "with a Jetson onnxruntime-gpu build that includes libonnxruntime_providers_tensorrt.so.");
      }
      if (!append_trt()) throw std::runtime_error("--ep tensorrt requested but TensorRT EP append failed");
      // Recommended provider chain: TensorRT first, CUDA second, CPU last.
      if (has_cuda) (void)append_cuda();
      return opt;
    }
    if (requested == "auto") {
      if (has_trt && append_trt()) {
        if (has_cuda) (void)append_cuda();
        return opt;
      }
      if (has_cuda && append_cuda()) return opt;
      std::cout << "[generic_composite_policy_player][omnixtreme] " << session_name << " auto selected CPUExecutionProvider\n";
      return opt;
    }
    throw std::runtime_error("unknown --ep/--execution-provider: " + args_.execution_provider);
  }

  std::array<float,4> run_fk_anchor(const std::vector<float>& q, const std::array<float,4>& base_quat_wxyz) {
    std::array<float, 3> ja{{q[12], q[13], q[14]}};
    std::array<float, 3> base_pos{{0.f, 0.f, 0.79f}};
    std::array<int64_t, 2> s13{{1,3}}, s14{{1,4}};
    std::array<const char*, 3> in_names{{"joint_angles", "base_pos", "base_quat"}};
    std::array<const char*, 2> out_names{{"pos", "rot"}};
    std::vector<float> bq(base_quat_wxyz.begin(), base_quat_wxyz.end());
    std::array<Ort::Value, 3> inputs{{
        Ort::Value::CreateTensor<float>(mem_, ja.data(), ja.size(), s13.data(), s13.size()),
        Ort::Value::CreateTensor<float>(mem_, base_pos.data(), base_pos.size(), s13.data(), s13.size()),
        Ort::Value::CreateTensor<float>(mem_, bq.data(), bq.size(), s14.data(), s14.size())}};
    auto outs = fk_session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names.data(), out_names.size());
    (void)tensor_to_vec_exact(outs.at(0), 3, "OmniXtreme FK pos");
    const auto rot = tensor_to_vec_exact(outs.at(1), 4, "OmniXtreme FK rot");
    const float* r = rot.data();
    return normalize_quat({r[0], r[1], r[2], r[3]});
  }

  std::vector<float> run_base(std::vector<float>& real, std::vector<float>& command, std::vector<float>& hist, std::vector<float>& noise) {
    std::array<int64_t, 2> s_real{{1, kRealObsDim}};
    std::array<int64_t, 2> s_cmd{{1, kCommandObsDim}};
    std::array<int64_t, 2> s_hist{{1, kHistDim}};
    std::array<int64_t, 2> s_noise{{1, kNumJoints}};
    std::array<const char*, 4> in_names{{"real_obs", "command_obs", "real_historical_obs_raw", "initial_noise"}};
    std::array<const char*, 1> out_names{{"action"}};
    std::array<Ort::Value, 4> inputs{{
        Ort::Value::CreateTensor<float>(mem_, real.data(), real.size(), s_real.data(), s_real.size()),
        Ort::Value::CreateTensor<float>(mem_, command.data(), command.size(), s_cmd.data(), s_cmd.size()),
        Ort::Value::CreateTensor<float>(mem_, hist.data(), hist.size(), s_hist.data(), s_hist.size()),
        Ort::Value::CreateTensor<float>(mem_, noise.data(), noise.size(), s_noise.data(), s_noise.size())}};
    auto outs = base_session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names.data(), out_names.size());
    return tensor_to_vec_exact(outs.at(0), kNumJoints, "OmniXtreme base action");
  }

  std::vector<float> run_residual(std::vector<float>& obs) {
    std::array<int64_t, 2> s_obs{{1, kResidualObsDim}};
    std::array<const char*, 1> in_names{{"obs"}};
    std::array<const char*, 1> out_names{{"actions"}};
    std::array<Ort::Value, 1> inputs{{
        Ort::Value::CreateTensor<float>(mem_, obs.data(), obs.size(), s_obs.data(), s_obs.size())}};
    auto outs = residual_session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names.data(), out_names.size());
    return tensor_to_vec_exact(outs.at(0), kNumJoints, "OmniXtreme residual actions");
  }

  void init_heading(const std::array<float,4>& robot_quat) {
    const float robot_yaw = quat_yaw(robot_quat);
    const float motion_yaw = quat_yaw(motion_.body_quat_at(args_.motion_start_frame, 0));
    yaw_offset_ = robot_yaw - motion_yaw;
    heading_initialized_ = true;
    std::cout << "[generic_composite_policy_player][omnixtreme] heading initialized robot_yaw=" << robot_yaw
              << " motion_yaw=" << motion_yaw << " yaw_offset=" << yaw_offset_ << "\n";
  }

  std::array<float,4> corrected_base_quat(const std::array<float,4>& robot_quat) const {
    const auto rpy = quat_rpy(robot_quat);
    return quat_from_rpy(rpy[0], rpy[1], rpy[2] - yaw_offset_);
  }

  void clip_with_envelope(const std::vector<float>& q, const std::vector<float>& dq, std::vector<float>& target) const {
    constexpr float eps = 1e-6f;
    constexpr float v_eps = 1e-2f;
    for (int i = 0; i < kNumJoints; ++i) {
      const float abs_dq = std::fabs(dq[i]);
      const float over = std::max(0.f, abs_dq - friction_x1_[static_cast<size_t>(i)]);
      const float base_pos = (abs_dq <= v_eps) ? friction_y2_[static_cast<size_t>(i)] : (dq[i] >= 0.f ? friction_y1_[static_cast<size_t>(i)] : friction_y2_[static_cast<size_t>(i)]);
      const float slope_pos = base_pos / std::max(eps, friction_x2_[static_cast<size_t>(i)] - friction_x1_[static_cast<size_t>(i)]);
      const float tau_high = std::max(0.f, base_pos - slope_pos * over);
      const float base_neg = (abs_dq <= v_eps) ? -friction_y2_[static_cast<size_t>(i)] : (dq[i] >= 0.f ? -friction_y2_[static_cast<size_t>(i)] : -friction_y1_[static_cast<size_t>(i)]);
      const float slope_neg = (-base_neg) / std::max(eps, friction_x2_[static_cast<size_t>(i)] - friction_x1_[static_cast<size_t>(i)]);
      const float tau_low = std::min(0.f, base_neg + slope_neg * over);
      const float p_low = (tau_low + kd_[i] * dq[i]) / std::max(eps, kp_[i]) + q[i];
      const float p_high = (tau_high + kd_[i] * dq[i]) / std::max(eps, kp_[i]) + q[i];
      target[i] = std::clamp(target[i], std::min(p_low, p_high), std::max(p_low, p_high));
    }
  }

  Args args_;
  MotionCache motion_;
  Ort::Env env_;
  Ort::Session base_session_{nullptr};
  Ort::Session residual_session_{nullptr};
  Ort::Session fk_session_{nullptr};
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<float> default_pos_, action_scale_, kp_, kd_, torque_limits_, lower_limits_, upper_limits_;
  std::vector<float> friction_x1_, friction_x2_, friction_y1_, friction_y2_, friction_va_, friction_fs_, friction_fd_;
  std::vector<int> joint_permutation_, inverse_joint_permutation_;
  std::vector<float> history_, prev_action_, prev_base_action_;
  std::mt19937 rng_;
  double residual_gain_{1.0};
  double initial_noise_scale_{1.0};
  float base_ang_vel_noise_{0.1f};
  float base_ang_vel_scale_{1.0f};
  int anchor_body_index_{0};
  int frame_{0};
  bool heading_initialized_{false};
  float yaw_offset_{0.f};
};

class UnitreeDDS {
public:
  explicit UnitreeDDS(const Args& args,
                      const std::vector<float>& lower_limits,
                      const std::vector<float>& upper_limits,
                      const std::vector<float>& torque_limits,
                      std::vector<float> friction_va,
                      std::vector<float> friction_fs,
                      std::vector<float> friction_fd)
      : args_(args),
        friction_va_(std::move(friction_va)),
        friction_fs_(std::move(friction_fs)),
        friction_fd_(std::move(friction_fd)),
        torque_limits_(torque_limits),
        activation_blend_(args.activation_blend_sec) {
    event_log_.open_append(args_.event_log.empty() ? "" : args_.event_log + ".dds.jsonl");
    yamlc::require_size("safety.friction.va", kNumJoints, friction_va_.size());
    yamlc::require_size("safety.friction.fs", kNumJoints, friction_fs_.size());
    yamlc::require_size("safety.friction.fd", kNumJoints, friction_fd_.size());
    CommandSafetyConfig safety_cfg;
    safety_cfg.joint_limit_enabled = args_.safety_joint_limit_enabled;
    safety_cfg.lowstate_timeout_sec = 0.2;
    safety_filter_.set_config(safety_cfg);
    safety_limits_.lower = lower_limits;
    safety_limits_.upper = upper_limits;
    unitree::robot::ChannelFactory::Instance()->Init(args_.domain_id, args_.net);
    publisher_.reset(new unitree::robot::ChannelPublisher<LowCmd>(args_.cmd_topic));
    publisher_->InitChannel();
    subscriber_.reset(new unitree::robot::ChannelSubscriber<LowState>(args_.state_topic));
    subscriber_->InitChannel(std::bind(&UnitreeDDS::on_low_state, this, std::placeholders::_1), 1);
  }

  void wait_for_state() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!connected_.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!connected_.load()) throw std::runtime_error("timeout waiting for lowstate");
    std::cout << "[generic_composite_policy_player][omnixtreme] lowstate connected\n";
  }

  OmniXtremePolicy::RobotState snapshot() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_;
  }

  uint64_t sequence() const { return sequence_.load(); }

  void publish(const std::vector<float>& q, const std::vector<float>& kp, const std::vector<float>& kd,
               bool allow_stale_state = false, bool active_command = true,
               bool bypass_activation = false,
               action_go::runtime::CommandTiming timing = {}, bool damping_command = false) {
    if (args_.dry_run) return;
    if (!active_command && !damping_command && args_.experimental_no_prestart_hold) {
      if (!prestart_hold_suppressed_logged_) {
        prestart_hold_suppressed_logged_ = true;
        event_log_.log("prestart_hold_suppressed",
                       {event_string("player", "generic_composite_policy_player"),
                        event_string("variant", "omnixtreme"),
                        event_string("reason", "experimental_handoff_ablation")});
      }
      return;
    }
    auto s = snapshot();
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
                       {event_string("player", "generic_composite_policy_player"),
                        event_string("variant", "omnixtreme"),
                        event_number("duration_sec", args_.activation_blend_sec),
                        event_number("max_q_delta", blend_result.max_q_delta),
                        event_number("max_kp_delta", blend_result.max_kp_delta),
                        event_number("max_kd_delta", blend_result.max_kd_delta)});
      }
      if (bypass_activation && !activation_blend_bypass_logged_) {
        activation_blend_bypass_logged_ = true;
        event_log_.log("activation_blend_bypassed",
                       {event_string("player", "generic_composite_policy_player"),
                        event_string("variant", "omnixtreme"),
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
                     {event_string("player", "generic_composite_policy_player"),
                      event_string("variant", "omnixtreme"),
                      event_string("reason", e.what()),
                      event_int("state_seq", static_cast<long long>(safety_state.seq))});
      std::cerr << "[generic_composite_policy_player][omnixtreme] safety stop: " << e.what() << "\n";
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
      float tau_ff = 0.f;
      if (args_.friction_feedforward && i < static_cast<int>(s.dq.size())) {
        // Match OmniXtreme/deploy_mujoco.py pd_control_friction:
        // applied = kp*(q_des-q) + kd*(0-dq) -
        //           (Fs*tanh(dq/Va) + Fd*dq).
        tau_ff = -(friction_fs_[static_cast<size_t>(i)] *
                   std::tanh(s.dq[i] / friction_va_[static_cast<size_t>(i)]) +
                   friction_fd_[static_cast<size_t>(i)] * s.dq[i]);
      }
      const float pd_torque = filtered.kp[i] * (filtered.q[i] - s.q[static_cast<size_t>(i)]) -
                              filtered.kd[i] * s.dq[static_cast<size_t>(i)];
      const float total_limit = torque_limits_[static_cast<size_t>(i)];
      tau_ff = std::clamp(pd_torque + tau_ff, -total_limit, total_limit) - pd_torque;
      cmd.motor_cmd()[i].tau() = tau_ff;
      cmd.motor_cmd()[i].kp() = filtered.kp[i];
      cmd.motor_cmd()[i].kd() = filtered.kd[i];
    }
    cmd.crc() = crc32_core(reinterpret_cast<uint32_t*>(&cmd), (sizeof(cmd) >> 2) - 1);
    if (!publisher_->Write(cmd)) {
      event_log_.log("publish_failed");
      return;
    }
    if (active_command) timing_reporter_.published(event_log_, timing);
    if (!active_command) {
      activation_blend_.observe_hold({filtered.q, filtered.kp, filtered.kd});
    } else if (blend_result.completed && !blend_result.bypassed &&
               args_.activation_blend_sec > 0.0) {
      event_log_.log("activation_blend_completed",
                     {event_string("player", "generic_composite_policy_player"),
                      event_string("variant", "omnixtreme"),
                      event_number("duration_sec", args_.activation_blend_sec)});
    }
    if (!first_command_logged_) {
      first_command_logged_ = true;
      event_log_.log("first_command_published",
                     {event_string("player", "generic_composite_policy_player"),
                      event_string("variant", "omnixtreme"),
                      event_string("command_phase", active_command ? "active" : "hold"),
                      event_int("state_seq", static_cast<long long>(safety_state.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
    if (!active_command && !first_hold_logged_) {
      first_hold_logged_ = true;
      event_log_.log("first_hold_published",
                     {event_string("player", "generic_composite_policy_player"),
                      event_string("variant", "omnixtreme"),
                      event_int("state_seq", static_cast<long long>(safety_state.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
    if (active_command && !first_active_command_logged_) {
      first_active_command_logged_ = true;
      event_log_.log("first_active_command",
                     {event_string("player", "generic_composite_policy_player"),
                      event_string("variant", "omnixtreme"),
                      event_int("state_seq", static_cast<long long>(safety_state.seq)),
                      event_int("mode_machine", s.mode_machine > 0 ? s.mode_machine : args_.mode_machine)});
    }
  }

  void publish_hold_current(const std::vector<float>& kp, const std::vector<float>& kd) {
    auto s = snapshot();
    if (s.q.size() < kNumJoints) return;
    publish(s.q, kp, kd, true, false);
  }

  void publish_damping_for_stop() {
    if (args_.dry_run || args_.no_damping_on_stop) return;
    auto s = snapshot();
    if (s.q.size() < kNumJoints) return;
    std::vector<float> kp(kNumJoints, 0.f), kd(kNumJoints, 3.f);
    for (int i = 0; i < 20; ++i) {
      publish(s.q, kp, kd, true, false, false, {}, true);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
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
    OmniXtremePolicy::RobotState next;
    next.mode_machine = static_cast<int>(low.mode_machine());
    next.quat = low.imu_state().quaternion();
    next.gyro = low.imu_state().gyroscope();
    next.q.assign(kNumJoints, 0.f);
    next.dq.assign(kNumJoints, 0.f);
    for (size_t i = 0; i < kNumJoints; ++i) {
      next.q[i] = low.motor_state()[i].q();
      next.dq[i] = low.motor_state()[i].dq();
      if (!finite(next.q[i]) || !finite(next.dq[i])) return;
    }
    next.recv_stamp = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mtx_);
    next.seq = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    state_ = std::move(next);
    connected_.store(true);
  }

  const Args& args_;
  unitree::robot::ChannelPublisherPtr<LowCmd> publisher_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  mutable std::mutex mtx_;
  OmniXtremePolicy::RobotState state_;
  std::vector<float> friction_va_, friction_fs_, friction_fd_;
  std::vector<float> torque_limits_;
  CommandSafetyFilter safety_filter_;
  CommandSafetyLimits safety_limits_;
  RuntimeEventLogger event_log_;
  action_go::runtime::CommandTimingReporter timing_reporter_;
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

int run_omnixtreme_runtime(int argc, char** argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    Args args = parse_args(argc, argv);
    RuntimeEventLogger event_log;
    event_log.open_append(args.event_log);
    require_real_asset_file(args.base_model, "base ONNX model");
    require_real_asset_file(args.residual_model, "residual ONNX model");
    require_real_asset_file(args.fk_model, "FK ONNX model");
    require_real_asset_file(args.motion, "motion npz");
    OmniXtremePolicy policy(args);
    event_log.log("model_loaded",
                  {event_string("player", "generic_composite_policy_player"),
                   event_string("variant", "omnixtreme"),
                   event_string("model", args.base_model)});
    if (args.check_only) {
      policy.check_only();
      event_log.log("check_only_ok",
                    {event_string("player", "generic_composite_policy_player"),
                     event_string("variant", "omnixtreme")});
      return 0;
    }

    UnitreeDDS dds(args,
                   policy.lower_limits(),
                   policy.upper_limits(),
                   policy.torque_limits(),
                   policy.friction_va(),
                   policy.friction_fs(),
                   policy.friction_fd());
    dds.wait_for_state();
    const int publish_hz = std::max(args.inference_hz, args.publish_hz);
    const int infer_interval = std::max(1, publish_hz / std::max(1, args.inference_hz));
    const auto tick = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / publish_hz));
    const auto& kp = policy.kp();
    const auto& kd = policy.kd();

    policy.warmup(dds.snapshot(), args.warmup_iters);
    dds.publish_hold_current(kp, kd);
    if (!args.ready_file.empty()) {
      touch_file(args.ready_file);
      event_log.log("ready_file_touched",
                    {event_string("player", "generic_composite_policy_player"),
                     event_string("variant", "omnixtreme"),
                     event_string("path", args.ready_file)});
      std::cout << "[generic_composite_policy_player][omnixtreme] ready file touched: " << args.ready_file << "\n";
    }
    const bool preview_mode = !args.preview_file.empty();
    bool preview_committed = false;
    std::vector<float> preview_candidate;
    if (preview_mode) {
      const auto saved_state = policy.snapshot_state();
      const auto preview_robot_state = dds.snapshot();
      const int preview_frame = policy.frame();
      preview_candidate = policy.infer(preview_robot_state);
      policy.advance(false);
      event_log.log("preview_input_snapshot",
                    {event_string("player", "generic_composite_policy_player"),
                     event_string("template", "composite_motion_policy"),
                     event_string("variant", "omnixtreme"),
                     event_string("source", "motion_phase_and_composite_state"),
                     event_int("input_sequence", preview_frame + 1),
                     event_number("input_freshness_sec", 0.0),
                     event_string("input_hash", hash_float_vector(preview_candidate))});
      const auto decision = action_go::runtime::wait_preview_lifecycle(
          args.preview_file, args.commit_file, args.discard_file,
          args.preview_timeout_sec, publish_hz,
          []() { return g_stop_requested.load(); },
          [&]() { dds.publish(preview_candidate, kp, kd, false, true, true); });
      if (decision != action_go::runtime::PreviewLifecycleDecision::kCommit) {
        policy.restore_state(saved_state);
        event_log.log("preview_state_restored",
                      {event_string("player", "generic_composite_policy_player"),
                       event_string("template", "composite_motion_policy"),
                       event_string("variant", "omnixtreme")});
        if (decision == action_go::runtime::PreviewLifecycleDecision::kTimeout) {
          throw std::runtime_error("transactional preview timed out before commit");
        }
        return 0;
      }
      preview_committed = true;
      event_log.log("preview_commit_first_command_consistent",
                    {event_string("player", "generic_composite_policy_player"),
                     event_string("variant", "omnixtreme"),
                     event_string("candidate_hash", hash_float_vector(preview_candidate))});
    } else if (!args.start_file.empty()) {
      std::cout << "[generic_composite_policy_player][omnixtreme] holding current pose until start file: " << args.start_file << "\n";
      auto hold_next = std::chrono::steady_clock::now();
      while (!g_stop_requested.load() && !fs::exists(args.start_file)) {
        dds.publish_hold_current(kp, kd);
        hold_next += tick;
        if (hold_next < std::chrono::steady_clock::now()) hold_next = std::chrono::steady_clock::now();
        std::this_thread::sleep_until(hold_next);
      }
      if (fs::exists(args.start_file)) {
        event_log.log("start_file_seen",
                      {event_string("player", "generic_composite_policy_player"),
                       event_string("variant", "omnixtreme"),
                       event_string("path", args.start_file)});
      }
    }
    if (g_stop_requested.load()) {
      dds.publish_damping_for_stop();
      if (dds.safety_failed()) throw std::runtime_error("CommandSafetyFilter stopped OmniXtreme publishing");
      return 0;
    }

    if (!preview_committed &&
        (args.prepare_ramp_sec > 1e-6 || args.prepare_hold_sec > 1e-6)) {
      const auto prep_q = policy.motion_start_q();
      auto prep_from = dds.snapshot().q;
      if (prep_from.size() < kNumJoints) prep_from = prep_q;
      prep_from.resize(kNumJoints, 0.f);

      const double ramp_sec = std::max(0.0, args.prepare_ramp_sec);
      const double hold_sec = std::max(0.0, args.prepare_hold_sec);
      const double total_sec = ramp_sec + hold_sec;
      std::cout << "[generic_composite_policy_player][omnixtreme] preparing motion start pose after UserCtrl"
                << " ramp=" << ramp_sec
                << " hold=" << hold_sec << " sec\n";
      auto prep_next = std::chrono::steady_clock::now();
      const auto prep_begin = prep_next;
      while (!g_stop_requested.load()) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - prep_begin).count();
        if (elapsed >= total_sec) break;
        const double alpha = ramp_sec > 1e-6 ? std::min(elapsed / ramp_sec, 1.0) : 1.0;
        dds.publish(lerp_vec(prep_from, prep_q, alpha), kp, kd);
        prep_next += tick;
        if (prep_next < std::chrono::steady_clock::now()) prep_next = std::chrono::steady_clock::now();
        std::this_thread::sleep_until(prep_next);
      }
      dds.publish(prep_q, kp, kd);
    }

    auto next = std::chrono::steady_clock::now();
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> q_cmd(kNumJoints, 0.f), q_from(kNumJoints, 0.f), q_to(kNumJoints, 0.f);
    auto s0 = dds.snapshot();
    if (preview_committed) q_cmd = preview_candidate;
    else if (s0.q.size() >= kNumJoints) q_cmd = s0.q;
    q_from = q_cmd;
    q_to = q_cmd;
    const std::vector<float> startup_q = q_cmd;
    auto target_set_time = start;
    int tick_count = preview_committed ? 1 : 0;
    int infer_count = preview_committed ? 1 : 0;
    bool max_frames_reached = false;
    uint64_t last_lowstate_seq = dds.sequence();

    std::cout << "[generic_composite_policy_player][omnixtreme] start base=" << args.base_model
              << " residual=" << args.residual_model
              << " motion=" << args.motion
              << " cmd_topic=" << args.cmd_topic
              << " hz=" << args.inference_hz
              << " publish_hz=" << publish_hz
              << " max_frames=" << args.max_frames
              << " deterministic=" << (args.deterministic ? "true" : "false")
              << " dry_run=" << (args.dry_run ? "true" : "false")
              << " mode=" << (args.sync_lowstate ? "sync-lowstate" : "decoupled-publish") << "\n";

    if (args.sync_lowstate) {
      // Simulation mode: keep the old lock-step behavior.  MuJoCo publishes one
      // LowState, the player computes/publishes one LowCmd, then MuJoCo steps.
      // This intentionally lets the simulator wait for ONNX inference.
      bool committed_candidate_pending = preview_committed;
      action_go::runtime::CommandTiming command_timing;
      while (!g_stop_requested.load()) {
        while (!g_stop_requested.load()) {
            const uint64_t seq = dds.sequence();
            if (seq != last_lowstate_seq) {
              last_lowstate_seq = seq;
              break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        const auto now = std::chrono::steady_clock::now();
        if (!committed_candidate_pending && tick_count % infer_interval == 0) {
          const auto s = dds.snapshot();
          q_from = q_to;
          action_go::runtime::begin_command_timing(s.recv_stamp, s.seq);
          q_to = policy.infer(s);
          command_timing = action_go::runtime::finish_command_timing();
          target_set_time = now;
          policy.advance(false);
          ++infer_count;
          if (args.max_frames > 0 && infer_count >= args.max_frames) max_frames_reached = true;
          if (args.stats_interval > 0 && infer_count % args.stats_interval == 0) {
            const auto [mn, mx] = std::minmax_element(q_to.begin(), q_to.end());
            std::cout << "[generic_composite_policy_player][omnixtreme] frame=" << policy.frame() << "/" << policy.total_frames()-1
                      << " target=[" << *mn << "," << *mx << "]\n";
            event_log.log("runtime_metrics",
                          {event_string("player", "generic_composite_policy_player"),
                           event_string("template", "composite_motion_policy"),
                           event_string("variant", "omnixtreme"),
                           event_int("frame", policy.frame()),
                           event_int("step", infer_count),
                           event_number("target_min", *mn),
                           event_number("target_max", *mx)});
          }
        }

        q_cmd = q_to;
        if (args.target_interp_sec > 1e-6) {
          const double a = std::chrono::duration<double>(now - target_set_time).count() / args.target_interp_sec;
          q_cmd = lerp_vec(q_from, q_to, a);
        }
        if (!preview_committed && args.startup_interp_sec > 1e-6 && startup_q.size() == q_cmd.size()) {
          const double elapsed = std::chrono::duration<double>(now - start).count();
          const double a = elapsed / args.startup_interp_sec;
          q_cmd = lerp_vec(startup_q, q_cmd, a);
        }
        dds.publish(q_cmd, kp, kd, false, true, preview_committed, command_timing);
        committed_candidate_pending = false;

        if (max_frames_reached) break;
        if (policy.done()) break;
        if (args.duration_sec > 0.0) {
          const double elapsed = static_cast<double>(tick_count + 1) / static_cast<double>(publish_hz);
          if (elapsed >= args.duration_sec) break;
        }
        ++tick_count;
      }
    } else {
      // Real robot mode: keep LowCmd publication at high rate even while ONNX
      // inference is running.  The previous single-thread loop blocked 500 Hz
      // publishing during every 50 Hz inference call; on hardware that looks
      // like periodic command dropouts, while lock-step simulation hides it.
      std::mutex cmd_mtx;
      std::mutex stats_mtx;
      std::atomic<bool> publish_stop{false};
      double publish_late_max_ms = 0.0;
      int publish_late_count = 0;
      int64_t publish_count = 0;
      double max_publish_interval_ms = 0.0;
      double e2e_ms_sum = 0.0;
      double e2e_ms_max = 0.0;
      double e2e_ms_last = 0.0;
      int e2e_count = 0;
      uint64_t target_seq = 0;
      std::chrono::steady_clock::time_point target_source_stamp = s0.recv_stamp;
      action_go::runtime::CommandTiming target_timing;

      auto command_at_locked = [&](const std::chrono::steady_clock::time_point& now) {
        std::vector<float> out = q_to;
        if (args.target_interp_sec > 1e-6) {
          const double a = std::chrono::duration<double>(now - target_set_time).count() / args.target_interp_sec;
          out = lerp_vec(q_from, q_to, a);
        }
        if (!preview_committed && args.startup_interp_sec > 1e-6 && startup_q.size() == out.size()) {
          const double elapsed = std::chrono::duration<double>(now - start).count();
          const double a = elapsed / args.startup_interp_sec;
          out = lerp_vec(startup_q, out, a);
        }
        return out;
      };

      std::thread publisher([&]() {
        auto pub_next = std::chrono::steady_clock::now();
        auto previous_publish = std::chrono::steady_clock::time_point{};
        uint64_t last_published_target_seq = std::numeric_limits<uint64_t>::max();
        while (!publish_stop.load() && !g_stop_requested.load()) {
          const auto now = std::chrono::steady_clock::now();
          uint64_t local_target_seq = 0;
          std::chrono::steady_clock::time_point local_source_stamp{};
          action_go::runtime::CommandTiming publish_timing;
          {
            std::lock_guard<std::mutex> lock(cmd_mtx);
            q_cmd = command_at_locked(now);
            local_target_seq = target_seq;
            local_source_stamp = target_source_stamp;
            publish_timing = target_timing;
          }
          dds.publish(q_cmd, kp, kd, false, true, preview_committed, publish_timing);
          const auto after_publish = std::chrono::steady_clock::now();
          {
            std::lock_guard<std::mutex> stats_lock(stats_mtx);
            if (previous_publish.time_since_epoch().count() != 0) {
              max_publish_interval_ms = std::max(
                  max_publish_interval_ms,
                  std::chrono::duration<double, std::milli>(
                      after_publish - previous_publish).count());
            }
            previous_publish = after_publish;
            ++publish_count;
          }
          if (local_target_seq != last_published_target_seq && local_source_stamp.time_since_epoch().count() != 0) {
            last_published_target_seq = local_target_seq;
            const double e2e_ms = std::chrono::duration<double, std::milli>(after_publish - local_source_stamp).count();
            std::lock_guard<std::mutex> stats_lock(stats_mtx);
            e2e_ms_last = e2e_ms;
            e2e_ms_sum += e2e_ms;
            e2e_ms_max = std::max(e2e_ms_max, e2e_ms);
            ++e2e_count;
          }

          pub_next += tick;
          const auto after = std::chrono::steady_clock::now();
          if (pub_next < after) {
            const double late_ms = std::chrono::duration<double, std::milli>(after - pub_next).count();
            {
              std::lock_guard<std::mutex> stats_lock(stats_mtx);
              publish_late_max_ms = std::max(publish_late_max_ms, late_ms);
              ++publish_late_count;
            }
            pub_next = after;
          }
          std::this_thread::sleep_until(pub_next);
        }
      });

      const auto infer_tick = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / std::max(1, args.inference_hz)));
      auto next_infer = std::chrono::steady_clock::now() +
          (preview_committed ? infer_tick : std::chrono::steady_clock::duration::zero());
      double infer_ms_sum = 0.0;
      double infer_ms_max = 0.0;
      double infer_ms_last = 0.0;
      int infer_late_count = 0;

      try {
        if (preview_committed) std::this_thread::sleep_until(next_infer);
        while (!g_stop_requested.load()) {
          const auto infer_begin = std::chrono::steady_clock::now();
          const auto s = dds.snapshot();
          action_go::runtime::begin_command_timing(s.recv_stamp, s.seq);
          const auto next_target = policy.infer(s);
          const auto infer_end = std::chrono::steady_clock::now();
          infer_ms_last = std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
          infer_ms_sum += infer_ms_last;
          infer_ms_max = std::max(infer_ms_max, infer_ms_last);

          {
            std::lock_guard<std::mutex> lock(cmd_mtx);
            q_from = command_at_locked(infer_end);
            q_to = next_target;
            target_set_time = infer_end;
            target_source_stamp = s.recv_stamp;
            target_timing = action_go::runtime::finish_command_timing();
            ++target_seq;
          }

          policy.advance(false);
          ++infer_count;
          if (args.max_frames > 0 && infer_count >= args.max_frames) max_frames_reached = true;
          if (args.stats_interval > 0 && infer_count % args.stats_interval == 0) {
            const auto [mn, mx] = std::minmax_element(q_to.begin(), q_to.end());
            const double infer_avg = infer_ms_sum / std::max(1, infer_count);
            double e2e_last_copy = 0.0, e2e_avg_copy = 0.0, e2e_max_copy = 0.0, pub_late_max_copy = 0.0;
            double max_publish_interval_copy = 0.0;
            int e2e_count_copy = 0, pub_late_count_copy = 0;
            int64_t publish_count_copy = 0;
            {
              std::lock_guard<std::mutex> stats_lock(stats_mtx);
              e2e_last_copy = e2e_ms_last;
              e2e_avg_copy = e2e_count > 0 ? (e2e_ms_sum / static_cast<double>(e2e_count)) : 0.0;
              e2e_max_copy = e2e_ms_max;
              e2e_count_copy = e2e_count;
              pub_late_count_copy = publish_late_count;
              pub_late_max_copy = publish_late_max_ms;
              publish_count_copy = publish_count;
              max_publish_interval_copy = max_publish_interval_ms;
            }
            std::cout << "[generic_composite_policy_player][omnixtreme] frame=" << policy.frame() << "/" << policy.total_frames()-1
                      << " target=[" << *mn << "," << *mx << "]"
                      << " infer_ms(last/avg/max)=" << infer_ms_last << "/" << infer_avg << "/" << infer_ms_max
                      << " e2e_ms(last/avg/max,n)=" << e2e_last_copy << "/" << e2e_avg_copy << "/" << e2e_max_copy << "/" << e2e_count_copy
                      << " infer_late=" << infer_late_count
                      << " pub_late=" << pub_late_count_copy
                      << " pub_late_max_ms=" << pub_late_max_copy << "\n";
            event_log.log("runtime_metrics",
                          {event_string("player", "generic_composite_policy_player"),
                           event_string("template", "composite_motion_policy"),
                           event_string("variant", "omnixtreme"),
                           event_int("frame", policy.frame()),
                           event_int("step", infer_count),
                           event_number("target_min", *mn),
                           event_number("target_max", *mx),
                           event_number("infer_ms_last", infer_ms_last),
                           event_number("infer_ms_avg", infer_avg),
                           event_number("infer_ms_max", infer_ms_max),
                           event_number("e2e_ms_last", e2e_last_copy),
                           event_number("e2e_ms_avg", e2e_avg_copy),
                           event_number("e2e_ms_max", e2e_max_copy),
                           event_int("e2e_count", e2e_count_copy),
                           event_int("policy_deadline_miss", infer_late_count),
                           event_int("publish_count", publish_count_copy),
                           event_int("publish_deadline_miss", pub_late_count_copy),
                           event_number("max_publish_interval_ms", max_publish_interval_copy),
                           event_number("pub_late_max_ms", pub_late_max_copy)});
          }

          if (max_frames_reached) break;
          if (policy.done()) break;
          if (args.duration_sec > 0.0) {
            const double elapsed = std::chrono::duration<double>(infer_end - start).count();
            if (elapsed >= args.duration_sec) break;
          }

          next_infer += infer_tick;
          const auto sleep_from = std::chrono::steady_clock::now();
          if (next_infer < sleep_from) {
            ++infer_late_count;
            next_infer = sleep_from;
          }
          std::this_thread::sleep_until(next_infer);
        }
      } catch (...) {
        publish_stop.store(true);
        if (publisher.joinable()) publisher.join();
        throw;
      }

      publish_stop.store(true);
      if (publisher.joinable()) publisher.join();
      {
        std::lock_guard<std::mutex> stats_lock(stats_mtx);
        event_log.log("runtime_metrics",
                      {event_string("player", "generic_composite_policy_player"),
                       event_string("template", "composite_motion_policy"),
                       event_string("variant", "omnixtreme"),
                       event_string("phase", "final"),
                       event_int("step", infer_count),
                       event_int("publish_count", publish_count),
                       event_int("policy_deadline_miss", infer_late_count),
                       event_int("publish_deadline_miss", publish_late_count),
                       event_number("max_publish_interval_ms", max_publish_interval_ms),
                       event_number("pub_late_max_ms", publish_late_max_ms)});
      }
    }
    if (g_stop_requested.load()) dds.publish_damping_for_stop();
    if (dds.safety_failed()) throw std::runtime_error("CommandSafetyFilter stopped OmniXtreme publishing");
    std::cout << "[generic_composite_policy_player][omnixtreme] completed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[generic_composite_policy_player][omnixtreme] ERROR: " << e.what() << std::endl;
    return 1;
  }
}
