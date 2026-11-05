#pragma once

#include "action_go/runtime/generic_policy_runtime.h"
#include "action_go/runtime/activation_blend.h"
#include "action_go/runtime/async_command_publisher.h"
#include "action_go/runtime/bfm_action.h"
#include "action_go/runtime/bfm_observation.h"
#include "action_go/runtime/pbfm_policy.h"
#include "action_go/runtime/bfm_context_stream.h"
#include "action_go/runtime/shared_memory_depth_stream.h"
#include "action_go/runtime/command_safety_filter.h"
#include "action_go/runtime/event_logger.h"
#include "action_go/runtime/input_freshness.h"
#include "action_go/runtime/npy_reader.h"
#include "action_go/runtime/onnx_runner.h"
#include "action_go/runtime/policy_contract.h"
#include "action_go/runtime/policy_protocol.h"
#include "action_go/runtime/reference_endpoint.h"
#include "action_go/runtime/sonic_planner.h"
#include "action_go/runtime/sonic_policy.h"
#include "action_go/runtime/supermimic_policy.h"
#include "action_go/runtime/allmimic_policy.h"
#include "action_go/runtime/mjlab_combat_policy.h"
#include "action_go/runtime/yaml_contract.h"

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
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef ACTION_GO_HAS_ZMQ
#include <zmq.h>
#endif

#ifdef ACTION_GO_HAS_MUJOCO
#include <mujoco/mujoco.h>
#endif

namespace fs = std::filesystem;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using action_go::runtime::CommandSafetyConfig;
using action_go::runtime::CommandSafetyFilter;
using action_go::runtime::CommandSafetyLimits;
using action_go::runtime::CommandSafetyState;
using action_go::runtime::FilteredCommand;
using action_go::runtime::ActivationBlend;
using action_go::runtime::JointCommand;
using action_go::runtime::GenericPlayerKind;
using action_go::runtime::GenericPolicyRuntimeScope;
using action_go::runtime::OnnxRunner;
using action_go::runtime::PolicyContract;
using action_go::runtime::PolicyProtocolConstraint;
using action_go::runtime::RealtimeInputStaleGate;
using action_go::runtime::RuntimeEventLogger;
using action_go::runtime::SampleFreshness;
using action_go::runtime::SonicObservationBuilder;
using action_go::runtime::SonicObservationLayout;
using action_go::runtime::SonicReferenceCache;
using action_go::runtime::SonicReferenceFrame;
using action_go::runtime::SonicRobotFrame;
using action_go::runtime::StaleBehavior;
using action_go::runtime::StaleDecisionAction;
using action_go::runtime::contract_for_template;
using action_go::runtime::event_bool;
using action_go::runtime::event_int;
using action_go::runtime::event_number;
using action_go::runtime::event_string;
using action_go::runtime::contract_for_variant;
using action_go::runtime::format_protocol_constraint_summary;
using action_go::runtime::format_supported_templates;
using action_go::runtime::format_supported_variants;
using action_go::runtime::format_supported_variants_for_template;
using action_go::runtime::kBeyondMimicProtocolRefAnchorIndex;
using action_go::runtime::kBeyondMimicProtocolRefBodyCount;
using action_go::runtime::kBfmProtocolHistorySteps;
using action_go::runtime::kHumanoidGptProtocolHmcpHeaderSize;
using action_go::runtime::kHumanoidGptProtocolKeypointCount;
using action_go::runtime::kHumanoidGptProtocolObsDim;
using action_go::runtime::kHumanoidGptProtocolQposDim;
using action_go::runtime::kHumanoidGptProtocolQvelDim;
using action_go::runtime::kTwistProtocolActionDim;
using action_go::runtime::kTwistProtocolMimicDim;
using action_go::runtime::kTwistProtocolWristDim;
using action_go::runtime::kUnitreeG1ProtocolMotorCount;
using action_go::runtime::player_binary_name;
using action_go::runtime::protocol_constraint_for_variant;
using action_go::runtime::protocol_names;
using action_go::runtime::variant_allowed_for_template;
namespace yamlc = action_go::runtime::yaml_contract;

namespace action_go::runtime::generic_policy_runtime {
inline std::atomic<bool> g_stop_requested{false};
inline void signal_handler(int) { g_stop_requested.store(true); }

inline GenericPolicyRuntimeScope g_runtime_scope{GenericPolicyRuntimeScope::kPolicy};

inline void set_runtime_scope(GenericPolicyRuntimeScope scope) {
  g_runtime_scope = scope;
}

inline bool is_motion_tracking_player() {
  return g_runtime_scope == GenericPolicyRuntimeScope::kMotionTracking;
}

inline GenericPlayerKind player_kind() {
  if (g_runtime_scope == GenericPolicyRuntimeScope::kSensor) return GenericPlayerKind::kSensor;
  return is_motion_tracking_player() ? GenericPlayerKind::kMotionTracking : GenericPlayerKind::kPolicy;
}

inline std::string player_label() {
  return player_binary_name(player_kind());
}

inline const char* player_log_prefix() {
  if (g_runtime_scope == GenericPolicyRuntimeScope::kSensor) return "[generic_sensor_policy_player]";
  return is_motion_tracking_player() ? "[generic_motion_tracking_player]" : "[generic_policy_player]";
}

struct TwistMimicSample {
  SampleFreshness freshness{SampleFreshness::kMissing};
  std::array<float, kTwistProtocolMimicDim> values{};
  uint64_t sequence{0};
  double source_timestamp_sec{std::numeric_limits<double>::quiet_NaN()};
  double age_sec{std::numeric_limits<double>::infinity()};
  std::string content_hash;
};

struct HumanoidGptQposSample {
  SampleFreshness freshness{SampleFreshness::kMissing};
  std::array<float, kHumanoidGptProtocolQposDim> qpos{};
  uint32_t seq{0};
  double send_ts{0.0};
  double age_sec{std::numeric_limits<double>::infinity()};
};

struct SonicReferenceSample {
  SampleFreshness freshness{SampleFreshness::kMissing};
  std::vector<SonicReferenceFrame> frames;
  uint64_t sequence{0};
  double age_sec{std::numeric_limits<double>::infinity()};
};

struct SonicPlannerSample {
  SampleFreshness freshness{SampleFreshness::kMissing};
  SonicPlannerCommand command;
  uint64_t sequence{0};
  double age_sec{std::numeric_limits<double>::infinity()};
};

struct SonicControlSample {
  bool present{false};
  SonicControlCommand command;
  uint64_t sequence{0};
  double age_sec{std::numeric_limits<double>::infinity()};
};

inline std::string float_content_hash(const float* values, size_t count) {
  uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const uint8_t*>(values);
  for (size_t i = 0; i < count * sizeof(float); ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

inline TwistMimicSample twist_sample_from_last(bool has_last,
                                        const std::array<float, kTwistProtocolMimicDim>& last,
                                        std::chrono::steady_clock::time_point last_time,
                                        uint64_t sequence,
                                        double stale_sec) {
  TwistMimicSample sample;
  if (!has_last) return sample;
  sample.values = last;
  sample.sequence = sequence;
  sample.age_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_time).count();
  sample.content_hash = float_content_hash(sample.values.data(), sample.values.size());
  sample.freshness = (stale_sec >= 0.0 && sample.age_sec > stale_sec) ? SampleFreshness::kStale
                                                                      : SampleFreshness::kFresh;
  return sample;
}

inline uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
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

inline std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

inline std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

inline std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

inline fs::path executable_path() {
  std::error_code ec;
  fs::path p = fs::read_symlink("/proc/self/exe", ec);
  if (!ec && !p.empty()) return p;
  return {};
}

inline fs::path action_go_root_from_exe() {
  fs::path exe = executable_path();
  if (exe.empty()) return fs::current_path();
  fs::path dir = exe.parent_path();
  if (dir.filename() == "build") return dir.parent_path();
  return dir.parent_path();
}

inline bool has_suffix(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::vector<float> extract_numbers(const std::string& text) {
  std::vector<float> nums;
  const char* p = text.c_str();
  while (*p) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(p, &end);
    if (end != p && errno != ERANGE && std::isfinite(v)) {
      nums.push_back(static_cast<float>(v));
      p = end;
    } else {
      ++p;
    }
  }
  return nums;
}

inline std::optional<std::string> bracket_payload_after_key(const std::string& text, const std::string& key) {
  const size_t key_pos = text.find(key);
  if (key_pos == std::string::npos) return std::nullopt;
  const size_t begin = text.find('[', key_pos);
  if (begin == std::string::npos) return std::nullopt;
  int depth = 0;
  for (size_t i = begin; i < text.size(); ++i) {
    if (text[i] == '[') ++depth;
    else if (text[i] == ']') {
      --depth;
      if (depth == 0) return text.substr(begin + 1, i - begin - 1);
    }
  }
  return std::nullopt;
}

inline std::optional<std::array<float, kTwistProtocolMimicDim>> parse_twist_mimic_packet(const std::string& packet) {
  std::vector<float> nums;
  if (auto payload = bracket_payload_after_key(packet, "mimic_obs")) {
    nums = extract_numbers(*payload);
  } else if (auto payload2 = bracket_payload_after_key(packet, "mimic")) {
    nums = extract_numbers(*payload2);
  } else {
    nums = extract_numbers(packet);
  }
  if (nums.size() != kTwistProtocolMimicDim) return std::nullopt;
  if (!std::all_of(nums.begin(), nums.end(), [](float value) { return std::isfinite(value); })) {
    return std::nullopt;
  }
  std::array<float, kTwistProtocolMimicDim> out{};
  std::copy(nums.begin(), nums.end(), out.begin());
  return out;
}

inline uint16_t le16(const uint8_t* p);
inline uint32_t le32(const uint8_t* p);
inline float le_float32(const uint8_t* p);
inline double le_float64(const uint8_t* p);

inline std::optional<HumanoidGptQposSample> parse_humanoid_gpt_hmcp_packet(const uint8_t* data, size_t size) {
  if (size < kHumanoidGptProtocolHmcpHeaderSize) return std::nullopt;
  if (std::memcmp(data, "HMCP", 4) != 0) return std::nullopt;
  const uint8_t version = data[4];
  if (version != 1) return std::nullopt;
  const uint8_t flags = data[5];
  const uint32_t seq = le32(data + 6);
  const double send_ts = le_float64(data + 10);
  const uint16_t n_qpos = le16(data + 18);
  if (n_qpos != kHumanoidGptProtocolQposDim) return std::nullopt;
  size_t expected = kHumanoidGptProtocolHmcpHeaderSize + static_cast<size_t>(n_qpos) * sizeof(float);
  if (flags & 0x01) expected += 4 * sizeof(float);
  if (flags & 0x02) expected += 24 * sizeof(float);
  if (size != expected) return std::nullopt;
  HumanoidGptQposSample out;
  out.freshness = SampleFreshness::kFresh;
  out.seq = seq;
  out.send_ts = send_ts;
  for (int i = 0; i < kHumanoidGptProtocolQposDim; ++i) {
    out.qpos[i] = le_float32(data + kHumanoidGptProtocolHmcpHeaderSize + static_cast<size_t>(i) * sizeof(float));
    if (!std::isfinite(out.qpos[i])) return std::nullopt;
  }
  if (!std::isfinite(out.send_ts)) return std::nullopt;
  return out;
}

inline std::optional<HumanoidGptQposSample> parse_humanoid_gpt_qpos_text_packet(const std::string& packet) {
  std::vector<float> nums;
  if (auto payload = bracket_payload_after_key(packet, "qpos")) {
    nums = extract_numbers(*payload);
  } else {
    nums = extract_numbers(packet);
  }
  if (nums.size() != kHumanoidGptProtocolQposDim) return std::nullopt;
  if (!std::all_of(nums.begin(), nums.end(), [](float value) { return std::isfinite(value); })) {
    return std::nullopt;
  }
  HumanoidGptQposSample out;
  out.freshness = SampleFreshness::kFresh;
  std::copy(nums.begin(), nums.end(), out.qpos.begin());
  return out;
}

inline std::vector<uint8_t> read_binary_file(const fs::path& path) {
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

inline uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

inline uint32_t le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t le64(const uint8_t* p) {
  uint64_t out = 0;
  for (int i = 0; i < 8; ++i) out |= static_cast<uint64_t>(p[i]) << (8 * i);
  return out;
}

inline float le_float32(const uint8_t* p) {
  uint32_t raw = le32(p);
  float out = 0.f;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

inline double le_float64(const uint8_t* p) {
  uint64_t raw = le64(p);
  double out = 0.0;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

inline float wrap_pi(float a) {
  return std::atan2(std::sin(a), std::cos(a));
}

inline std::vector<uint8_t> inflate_raw_deflate(const uint8_t* src, size_t src_size, size_t dst_size) {
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
  if (ret != Z_STREAM_END || zs.total_out != dst_size) {
    throw std::runtime_error("zip deflate inflate failed");
  }
  return dst;
}

inline std::unordered_map<std::string, std::vector<uint8_t>> read_npz_entries(const fs::path& path) {
  const auto bytes = read_binary_file(path);
  std::unordered_map<std::string, std::vector<uint8_t>> entries;
  size_t pos = 0;
  while (pos + 30 <= bytes.size()) {
    const uint32_t sig = le32(bytes.data() + pos);
    if (sig == 0x02014b50 || sig == 0x06054b50) break;
    if (sig != 0x04034b50) throw std::runtime_error("invalid zip local header in " + path.string());
    const uint16_t flags = le16(bytes.data() + pos + 6);
    const uint16_t method = le16(bytes.data() + pos + 8);
    const uint32_t comp_size32 = le32(bytes.data() + pos + 18);
    const uint32_t uncomp_size32 = le32(bytes.data() + pos + 22);
    const uint16_t name_len = le16(bytes.data() + pos + 26);
    const uint16_t extra_len = le16(bytes.data() + pos + 28);
    if (flags & 0x08) throw std::runtime_error("npz uses data descriptors; unsupported: " + path.string());
    const size_t name_pos = pos + 30;
    const size_t data_pos = name_pos + name_len + extra_len;
    size_t comp_size = comp_size32;
    size_t uncomp_size = uncomp_size32;
    if (comp_size32 == 0xFFFFFFFFu || uncomp_size32 == 0xFFFFFFFFu) {
      size_t ep = name_pos + name_len;
      const size_t extra_end = ep + extra_len;
      bool found_zip64 = false;
      while (ep + 4 <= extra_end) {
        const uint16_t header_id = le16(bytes.data() + ep);
        const uint16_t data_size = le16(bytes.data() + ep + 2);
        ep += 4;
        if (ep + data_size > extra_end) break;
        if (header_id == 0x0001) {
          size_t cursor = ep;
          if (uncomp_size32 == 0xFFFFFFFFu) {
            if (cursor + 8 > ep + data_size) throw std::runtime_error("bad ZIP64 uncompressed size in " + path.string());
            uncomp_size = static_cast<size_t>(le64(bytes.data() + cursor));
            cursor += 8;
          }
          if (comp_size32 == 0xFFFFFFFFu) {
            if (cursor + 8 > ep + data_size) throw std::runtime_error("bad ZIP64 compressed size in " + path.string());
            comp_size = static_cast<size_t>(le64(bytes.data() + cursor));
          }
          found_zip64 = true;
          break;
        }
        ep += data_size;
      }
      if (!found_zip64) throw std::runtime_error("missing ZIP64 size extra field in " + path.string());
    }
    if (data_pos + comp_size > bytes.size()) throw std::runtime_error("truncated zip entry");
    std::string name(reinterpret_cast<const char*>(bytes.data() + name_pos), name_len);
    std::vector<uint8_t> payload;
    if (method == 0) {
      payload.assign(bytes.begin() + data_pos, bytes.begin() + data_pos + comp_size);
    } else if (method == 8) {
      payload = inflate_raw_deflate(bytes.data() + data_pos, comp_size, uncomp_size);
    } else {
      throw std::runtime_error("unsupported npz compression method " + std::to_string(method));
    }
    if (has_suffix(name, ".npy")) name = name.substr(0, name.size() - 4);
    entries[name] = std::move(payload);
    pos = data_pos + comp_size;
    if (pos + 4 <= bytes.size()) {
      auto known_zip_sig = [&](size_t p) {
        const uint32_t next_sig = le32(bytes.data() + p);
        return next_sig == 0x04034b50 || next_sig == 0x02014b50 || next_sig == 0x06054b50;
      };
      if (!known_zip_sig(pos)) {
        size_t scan = pos + 1;
        const size_t scan_end = std::min(bytes.size(), pos + 64);
        while (scan + 4 <= scan_end && !known_zip_sig(scan)) ++scan;
        if (scan + 4 <= scan_end && known_zip_sig(scan)) pos = scan;
      }
    }
  }
  if (entries.empty()) throw std::runtime_error("no npy entries in npz: " + path.string());
  return entries;
}

inline std::vector<float> npy_to_float_vector(const std::vector<uint8_t>& bytes, const std::string& name) {
  return action_go::runtime::npy::to_float_vector(bytes, name);
}

inline int64_t npy_to_int64_scalar(const std::vector<uint8_t>& bytes, const std::string& name) {
  return action_go::runtime::npy::to_int64_scalar(bytes, name);
}

inline float npy_to_float_scalar(const std::vector<uint8_t>& bytes, const std::string& name) {
  return action_go::runtime::npy::to_float_scalar(bytes, name);
}

inline action_go::runtime::npy::ArrayView npy_meta(
    const std::unordered_map<std::string, std::vector<uint8_t>>& npz,
    const std::string& key) {
  return action_go::runtime::npy::metadata(npz, key);
}

inline std::string first_npz_key(const std::unordered_map<std::string, std::vector<uint8_t>>& npz,
                          std::initializer_list<const char*> keys,
                          const std::string& semantic) {
  for (const char* key : keys) {
    if (npz.find(key) != npz.end()) return key;
  }
  throw std::runtime_error("missing npz entry for " + semantic);
}

inline void check_size(const std::string& name, size_t expect, size_t actual) {
  if (expect != actual) {
    throw std::runtime_error(name + " size mismatch expect=" + std::to_string(expect) +
                             " actual=" + std::to_string(actual));
  }
}

template <size_t N>
std::array<float, N> to_float_array(const std::vector<float>& values, const std::string& name) {
  check_size(name, N, values.size());
  std::array<float, N> out{};
  std::copy(values.begin(), values.end(), out.begin());
  return out;
}

template <typename T>
T scalar_or(const YAML::Node& node, const std::string& key, const T& def) {
  if (!node || !node[key]) return def;
  return node[key].as<T>();
}

inline std::optional<int> int_field(const YAML::Node& node, const std::string& key) {
  if (!node || !node[key]) return std::nullopt;
  if (node[key].IsScalar()) {
    const std::string raw = lower(node[key].as<std::string>());
    if (raw.empty() || raw == "auto") return std::nullopt;
  }
  return node[key].as<int>();
}

inline int int_or(const YAML::Node& node, const std::string& key, int fallback) {
  if (auto value = int_field(node, key)) return *value;
  return fallback;
}

template <typename T>
std::vector<T> read_vec(const YAML::Node& root, const std::string& key) {
  if (!root[key]) throw std::runtime_error("missing params key: " + key);
  std::vector<T> out;
  for (const auto& v : root[key]) out.push_back(v.as<T>());
  return out;
}

template <typename T>
std::vector<T> read_vec_or(const YAML::Node& root, const std::string& key, const std::vector<T>& def) {
  if (!root || !root[key]) return def;
  return read_vec<T>(root, key);
}

inline std::vector<YAML::Node> sequence_nodes(const YAML::Node& node) {
  std::vector<YAML::Node> out;
  if (!node) return out;
  if (node.IsSequence()) {
    for (const auto& v : node) out.push_back(v);
  } else {
    out.push_back(node);
  }
  return out;
}

inline std::string node_scalar_string(const YAML::Node& node) {
  if (!node) return {};
  try { return node.as<std::string>(); } catch (...) { return {}; }
}

inline float node_float(const YAML::Node& node, float def = 0.f) {
  if (!node) return def;
  try { return node.as<float>(); } catch (...) { return def; }
}

inline std::vector<float> apply_regex_map(const YAML::Node& map, const std::vector<std::string>& joint_names, float def = 0.f) {
  std::vector<float> out(joint_names.size(), def);
  if (!map || !map.IsMap()) return out;
  for (const auto& kv : map) {
    const std::string pattern = kv.first.as<std::string>();
    const float value = node_float(kv.second, def);
    std::regex re(pattern);
    for (size_t i = 0; i < joint_names.size(); ++i) {
      if (std::regex_search(joint_names[i], re)) out[i] = value;
    }
  }
  return out;
}

inline std::vector<float> apply_actuator_field(const YAML::Node& actuators,
                                        const std::vector<std::string>& joint_names,
                                        const std::string& field,
                                        float def) {
  std::vector<float> out(joint_names.size(), def);
  if (!actuators) return out;
  for (const auto& actuator : sequence_nodes(actuators)) {
    if (!actuator || !actuator.IsMap() || !actuator[field]) continue;
    const float value = node_float(actuator[field], def);
    YAML::Node patterns = actuator["target_names_expr"];
    if (!patterns) patterns = actuator["actuator_names"];
    if (!patterns) patterns = actuator["joint_names"];
    for (const auto& pnode : sequence_nodes(patterns)) {
      const std::string pattern = node_scalar_string(pnode);
      if (pattern.empty()) continue;
      std::regex re(pattern);
      for (size_t i = 0; i < joint_names.size(); ++i) {
        if (std::regex_search(joint_names[i], re)) out[i] = value;
      }
    }
  }
  return out;
}

inline fs::path resolve_relative(const fs::path& base_file, const std::string& raw) {
  if (raw.empty()) return {};
  fs::path p(raw);
  if (p.is_absolute()) return p.lexically_normal();
  if (fs::exists(p)) return fs::absolute(p).lexically_normal();
  return (base_file.parent_path() / p).lexically_normal();
}

inline std::vector<float> read_float_file(const fs::path& path, size_t expected) {
  std::ifstream in(path);
  if (!in.good()) throw std::runtime_error("failed to open float file: " + path.string());
  std::vector<float> values;
  std::string token;
  while (in >> token) {
    for (char& c : token) {
      if (c == ',' || c == '[' || c == ']') c = ' ';
    }
    size_t pos = 0;
    while (pos < token.size()) {
      size_t consumed = 0;
      try {
        float v = std::stof(token.substr(pos), &consumed);
        values.push_back(v);
        pos += consumed;
      } catch (const std::invalid_argument&) {
        break;
      }
      while (pos < token.size() && std::isspace(static_cast<unsigned char>(token[pos]))) ++pos;
    }
  }
  check_size(path.string(), expected, values.size());
  return values;
}

inline std::array<float, 4> normalize_quat(std::array<float, 4> q) {
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(n > 1e-6f) || !std::isfinite(n)) return {1.f, 0.f, 0.f, 0.f};
  for (float& v : q) v /= n;
  return q;
}

inline std::array<float, 4> quat_mul(const std::array<float, 4>& a_in, const std::array<float, 4>& b_in) {
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

inline std::array<float, 4> quat_inverse(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  return {q[0], -q[1], -q[2], -q[3]};
}

inline std::array<float, 4> quat_mul_legacy(const std::array<float, 4>& q1, const std::array<float, 4>& q2) {
  const float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const float w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
  const float ww = (z1 + x1) * (x2 + y2);
  const float yy = (w1 - y1) * (w2 + z2);
  const float zz = (w1 + y1) * (w2 - z2);
  const float xx = ww + yy + zz;
  const float qq = 0.5f * (xx + (z1 - x1) * (x2 - y2));
  return {qq - ww + (z1 - y1) * (y2 - z2), qq - xx + (x1 + w1) * (x2 + w2),
          qq - yy + (w1 - x1) * (y2 + z2), qq - zz + (z1 + y1) * (w2 - x2)};
}

inline std::array<float, 4> axis_quat(float angle, char axis) {
  const float h = 0.5f * angle;
  const float c = std::cos(h);
  const float s = std::sin(h);
  if (axis == 'x') return {c, s, 0.f, 0.f};
  if (axis == 'y') return {c, 0.f, s, 0.f};
  return {c, 0.f, 0.f, s};
}

inline float quat_yaw(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  return std::atan2(2.f * (w * z + x * y), 1.f - 2.f * (y * y + z * z));
}

inline std::array<float, 4> yaw_quat(const std::array<float, 4>& q) {
  const float yaw = quat_yaw(q);
  return {std::cos(yaw * 0.5f), 0.f, 0.f, std::sin(yaw * 0.5f)};
}

inline std::array<float, 4> xyzw_to_wxyz(const std::array<float, 4>& q) {
  return normalize_quat({q[3], q[0], q[1], q[2]});
}

inline std::array<float, 4> normalize_quat_xyzw(std::array<float, 4> q) {
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(n > 1e-6f) || !std::isfinite(n)) return {0.f, 0.f, 0.f, 1.f};
  for (float& v : q) v /= n;
  return q;
}

inline std::array<float, 4> wxyz_to_xyzw(const std::array<float, 4>& q) {
  return normalize_quat_xyzw({q[1], q[2], q[3], q[0]});
}

inline std::array<float, 4> quat_mul_xyzw(const std::array<float, 4>& a_in, const std::array<float, 4>& b_in) {
  const auto a = normalize_quat_xyzw(a_in);
  const auto b = normalize_quat_xyzw(b_in);
  const float ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const float bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return normalize_quat_xyzw({
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
  });
}

inline std::array<float, 4> quat_conj_xyzw(std::array<float, 4> q) {
  q[0] *= -1.f;
  q[1] *= -1.f;
  q[2] *= -1.f;
  return q;
}

inline std::array<float, 4> axis_quat_xyzw(float angle, char axis) {
  const float h = 0.5f * angle;
  const float s = std::sin(h);
  const float c = std::cos(h);
  if (axis == 'x') return {s, 0.f, 0.f, c};
  if (axis == 'y') return {0.f, s, 0.f, c};
  return {0.f, 0.f, s, c};
}

inline std::array<float, 4> yaw_quat_xyzw(std::array<float, 4> q) {
  q = normalize_quat_xyzw(q);
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  const float yaw = std::atan2(2.f * (w * z + x * y), 1.f - 2.f * (y * y + z * z));
  return {0.f, 0.f, std::sin(0.5f * yaw), std::cos(0.5f * yaw)};
}

inline std::array<float, 4> compute_yaw_offset_xyzw(const std::array<float, 4>& robot_q,
                                             const std::array<float, 4>& motion_q) {
  return normalize_quat_xyzw(quat_mul_xyzw(yaw_quat_xyzw(robot_q), quat_conj_xyzw(yaw_quat_xyzw(motion_q))));
}

inline std::array<float, 4> remove_yaw_offset(const std::array<float, 4>& q, float yaw_offset) {
  if (std::abs(yaw_offset) < 1e-6f) return normalize_quat(q);
  return quat_mul(axis_quat(-yaw_offset, 'z'), q);
}

inline std::array<float, 3> quat_rpy(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  const float roll = std::atan2(2.f * (w * x + y * z), 1.f - 2.f * (x * x + y * y));
  const float sinp = 2.f * (w * y - z * x);
  const float pitch = std::asin(std::clamp(sinp, -1.f, 1.f));
  const float yaw = std::atan2(2.f * (w * z + x * y), 1.f - 2.f * (y * y + z * z));
  return {roll, pitch, yaw};
}

inline std::array<float, 9> matrix_from_quat(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  return {
      1.f - 2.f * (y * y + z * z), 2.f * (x * y - z * w),       2.f * (x * z + y * w),
      2.f * (x * y + z * w),       1.f - 2.f * (x * x + z * z), 2.f * (y * z - x * w),
      2.f * (x * z - y * w),       2.f * (y * z + x * w),       1.f - 2.f * (x * x + y * y)};
}

inline std::array<float, 16> identity_pose4() {
  return {1.f, 0.f, 0.f, 0.f,
          0.f, 1.f, 0.f, 0.f,
          0.f, 0.f, 1.f, 0.f,
          0.f, 0.f, 0.f, 1.f};
}

inline std::array<float, 16> pose_inverse(const std::array<float, 16>& T) {
  std::array<float, 16> out = identity_pose4();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out[r * 4 + c] = T[c * 4 + r];
  }
  for (int r = 0; r < 3; ++r) {
    out[r * 4 + 3] = -(out[r * 4 + 0] * T[3] + out[r * 4 + 1] * T[7] + out[r * 4 + 2] * T[11]);
  }
  return out;
}

inline std::array<float, 16> pose_mul(const std::array<float, 16>& A, const std::array<float, 16>& B) {
  std::array<float, 16> C{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      C[r * 4 + c] = A[r * 4 + 0] * B[0 * 4 + c] + A[r * 4 + 1] * B[1 * 4 + c] +
                     A[r * 4 + 2] * B[2 * 4 + c] + A[r * 4 + 3] * B[3 * 4 + c];
    }
  }
  return C;
}

inline std::array<float, 3> rot_apply(const std::array<float, 16>& T, const std::array<float, 3>& v) {
  return {
      T[0] * v[0] + T[1] * v[1] + T[2] * v[2],
      T[4] * v[0] + T[5] * v[1] + T[6] * v[2],
      T[8] * v[0] + T[9] * v[1] + T[10] * v[2],
  };
}

inline std::array<float, 16> navigation_pose_from_qpos(const std::array<float, kHumanoidGptProtocolQposDim>& qpos) {
  const auto R = matrix_from_quat({qpos[3], qpos[4], qpos[5], qpos[6]});
  const float r00 = R[0];
  const float r10 = R[3];
  const float norm = std::sqrt(r00 * r00 + r10 * r10);
  const float inv = 1.f / std::max(norm, 1e-12f);
  const float cx = r00 * inv;
  const float sx = r10 * inv;
  std::array<float, 16> T = identity_pose4();
  T[0] = cx;  T[1] = -sx; T[2] = 0.f; T[3] = qpos[0];
  T[4] = sx;  T[5] = cx;  T[6] = 0.f; T[7] = qpos[1];
  T[8] = 0.f; T[9] = 0.f; T[10] = 1.f; T[11] = 0.f;
  return T;
}

inline std::array<float, 3> rot_log_from_delta(const std::array<float, 16>& prev,
                                        const std::array<float, 16>& curr) {
  std::array<float, 9> R0T_R1{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R0T_R1[r * 3 + c] = prev[0 * 4 + r] * curr[0 * 4 + c] +
                           prev[1 * 4 + r] * curr[1 * 4 + c] +
                           prev[2 * 4 + r] * curr[2 * 4 + c];
    }
  }
  const float tr = R0T_R1[0] + R0T_R1[4] + R0T_R1[8];
  const float cos_theta = std::clamp((tr - 1.f) * 0.5f, -1.f, 1.f);
  const float theta = std::acos(cos_theta);
  const float safe = std::abs(theta) < 1e-6f ? 1.f : theta;
  const float k = std::abs(theta) < 1e-6f ? 0.5f : safe / (2.f * std::sin(safe));
  const std::array<float, 3> body{
      k * (R0T_R1[7] - R0T_R1[5]),
      k * (R0T_R1[2] - R0T_R1[6]),
      k * (R0T_R1[3] - R0T_R1[1]),
  };
  return {
      prev[0] * body[0] + prev[1] * body[1] + prev[2] * body[2],
      prev[4] * body[0] + prev[5] * body[1] + prev[6] * body[2],
      prev[8] * body[0] + prev[9] * body[1] + prev[10] * body[2],
  };
}

inline std::array<float, 9> mat_mul(const std::array<float, 9>& A, const std::array<float, 9>& B) {
  std::array<float, 9> C{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      C[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
    }
  }
  return C;
}

inline std::array<float, 9> mat_transpose(const std::array<float, 9>& A) {
  return {A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]};
}

inline std::array<float, 3> projected_gravity_from_quat(const std::array<float, 4>& q_in) {
  const auto q = normalize_quat(q_in);
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  const float r20 = 2.f * (x * z - y * w);
  const float r21 = 2.f * (y * z + x * w);
  const float r22 = 1.f - 2.f * (x * x + y * y);
  return {-r20, -r21, -r22};
}

inline float tilt_deg_from_quat(const std::array<float, 4>& q) {
  const auto g = projected_gravity_from_quat(q);
  const float c = std::clamp(-g[2], -1.f, 1.f);
  return std::acos(c) * 180.f / static_cast<float>(M_PI);
}

inline void append(std::vector<float>& dst, const std::vector<float>& src) { dst.insert(dst.end(), src.begin(), src.end()); }
inline void append(std::vector<float>& dst, const std::array<float, 3>& src) { dst.insert(dst.end(), src.begin(), src.end()); }

inline std::vector<float> lerp_vec(const std::vector<float>& a, const std::vector<float>& b, double alpha) {
  if (a.size() != b.size()) return b;
  alpha = std::clamp(alpha, 0.0, 1.0);
  std::vector<float> out(a.size(), 0.f);
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = static_cast<float>(static_cast<double>(a[i]) + alpha * (static_cast<double>(b[i]) - static_cast<double>(a[i])));
  }
  return out;
}

inline void push_history(std::vector<float>& history, const std::vector<float>& value, int dim, int steps) {
  check_size("history value", static_cast<size_t>(dim), value.size());
  check_size("history buffer", static_cast<size_t>(dim * steps), history.size());
  for (int step = steps - 1; step > 0; --step) {
    std::copy(history.begin() + (step - 1) * dim,
              history.begin() + step * dim,
              history.begin() + step * dim);
  }
  std::copy(value.begin(), value.end(), history.begin());
}

inline float sanitize_axis(float v) {
  if (!std::isfinite(v)) return 0.f;
  return std::clamp(v, -1.f, 1.f);
}

inline float unpack_float_le(const std::array<uint8_t, 40>& data, size_t begin) {
  if (begin + sizeof(float) > data.size()) return 0.f;
  float out = 0.f;
  std::memcpy(&out, data.data() + begin, sizeof(float));
  if (!std::isfinite(out)) return 0.f;
  return out;
}

inline float apply_deadband(float axis, float deadband, float positive_scale, float negative_scale) {
  axis = sanitize_axis(axis);
  deadband = std::clamp(deadband, 0.f, 0.95f);
  if (axis > deadband) return (axis - deadband) / std::max(1e-6f, 1.f - deadband) * positive_scale;
  if (axis < -deadband) return (axis + deadband) / std::max(1e-6f, 1.f - deadband) * negative_scale;
  return 0.f;
}

inline fs::path resolve_onnx_model(const fs::path& raw_model) {
  fs::path model = fs::absolute(raw_model).lexically_normal();
  const std::string ext = lower(model.extension().string());
  if (ext == ".onnx") return model;
  if (ext != ".pt" && ext != ".pth") return model;

  const fs::path parent = model.parent_path();
  std::vector<fs::path> candidates;
  if (!parent.filename().empty()) candidates.push_back(parent / (parent.filename().string() + ".onnx"));
  candidates.push_back(parent / "policy.onnx");
  candidates.push_back(parent / "exported.onnx");
  for (const auto& c : candidates) {
    if (fs::exists(c)) {
      std::cerr << player_log_prefix() << " C++ runtime received PyTorch checkpoint " << model
                << "; using adjacent ONNX " << c << "\n";
      return fs::absolute(c).lexically_normal();
    }
  }
  for (const auto& entry : fs::directory_iterator(parent)) {
    if (entry.is_regular_file() && lower(entry.path().extension().string()) == ".onnx") {
      std::cerr << player_log_prefix() << " C++ runtime received PyTorch checkpoint " << model
                << "; using first adjacent ONNX " << entry.path() << "\n";
      return fs::absolute(entry.path()).lexically_normal();
    }
  }
  throw std::runtime_error(std::string("C++ ") + player_label() +
                           " cannot run .pt directly; no adjacent ONNX found next to " + model.string());
}

struct Args {
  std::string spec;
  std::string net;
  int domain_id{0};
  std::string model_override;
  std::string params_override;
  std::string trajectory_override;
  std::string latent_file_override;
  std::string input_override;
  std::string motion_pkl_override;
  std::string motion_cache_override;
  std::string twist_root_override;
  std::string prepare_script_override;
  std::string steps_override;
  std::string realtime_transport_override;
  std::string udp_bind_override;
  std::string zmq_endpoint_override;
  std::string zmq_topic_override;
  std::string redis_host_override;
  std::string redis_key_override;
  std::string stale_behavior_override;
  std::string event_log;
  std::string execution_provider_override;
  std::string trt_cache_dir_override;
  std::string ready_file;
  std::string start_file;
  std::string handoff_complete_file;
  std::string handoff_release_file;
  std::string preview_file;
  std::string commit_file;
  std::string discard_file;
  std::string cmd_topic;
  std::string state_topic;
  std::string depth_source_override;
  std::string depth_shm_override;
  std::string trace_file;
  int inference_hz{0};
  int publish_hz{0};
  int sim_lowstate_decimation{0};
  int onnx_threads{0};
  int max_frames{-1};
  int udp_port{0};
  int redis_port{0};
  int gpu_device_id{-1};
  int warmup_iters{-1};
  double duration_sec{-1.0};
  double preview_timeout_sec{5.0};
  double input_timeout_sec{-1.0};
  double stale_sec{-1.0};
  double lowstate_timeout_sec{-1.0};
  double activation_blend_sec{std::numeric_limits<double>::quiet_NaN()};
  uint64_t trt_workspace_size{0};
  float action_scale_override{std::numeric_limits<float>::quiet_NaN()};
  bool rebuild_motion_cache{false};
  bool loop{false};
  bool require_realtime{false};
  bool stale_to_default{false};
  bool zmq_bind{true};
  bool zmq_bind_set{false};
  bool dry_run{false};
  bool dry_run_latency{false};
  bool check_only{false};
  bool damping_on_stop{true};
  bool experimental_no_prestart_hold{false};
  bool offline_dry_run_state{false};
  bool trt_fp16{false};
  bool trt_fp16_set{false};
};

inline std::string need_arg(int& i, int argc, char** argv, const std::string& name) {
  if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
  return argv[++i];
}

inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--spec") a.spec = need_arg(i, argc, argv, k);
    else if (k == "--net") a.net = need_arg(i, argc, argv, k);
    else if (k == "--domain-id") a.domain_id = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--model") a.model_override = need_arg(i, argc, argv, k);
    else if (k == "--params") a.params_override = need_arg(i, argc, argv, k);
    else if (k == "--depth-source") a.depth_source_override = need_arg(i, argc, argv, k);
    else if (k == "--depth-shm-name") a.depth_shm_override = need_arg(i, argc, argv, k);
    else if (k == "--trace-file") a.trace_file = need_arg(i, argc, argv, k);
    else if (k == "--trajectory" || k == "--motion") a.trajectory_override = need_arg(i, argc, argv, k);
    else if (k == "--latent-file" || k == "--z") a.latent_file_override = need_arg(i, argc, argv, k);
    else if (k == "--input") a.input_override = lower(need_arg(i, argc, argv, k));
    else if (k == "--motion-pkl") a.motion_pkl_override = need_arg(i, argc, argv, k);
    else if (k == "--motion-cache" || k == "--cache") a.motion_cache_override = need_arg(i, argc, argv, k);
    else if (k == "--twist-root") a.twist_root_override = need_arg(i, argc, argv, k);
    else if (k == "--prepare-script") a.prepare_script_override = need_arg(i, argc, argv, k);
    else if (k == "--steps") a.steps_override = need_arg(i, argc, argv, k);
    else if (k == "--rebuild-motion-cache") a.rebuild_motion_cache = true;
    else if (k == "--loop") a.loop = true;
    else if (k == "--require-realtime" || k == "--require-input") a.require_realtime = true;
    else if (k == "--stale-to-default") a.stale_to_default = true;
    else if (k == "--realtime-transport" || k == "--transport") a.realtime_transport_override = lower(need_arg(i, argc, argv, k));
    else if (k == "--udp-bind" || k == "--nokov-bind") a.udp_bind_override = need_arg(i, argc, argv, k);
    else if (k == "--udp-port" || k == "--nokov-port") a.udp_port = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--zmq-endpoint") a.zmq_endpoint_override = need_arg(i, argc, argv, k);
    else if (k == "--zmq-topic") a.zmq_topic_override = need_arg(i, argc, argv, k);
    else if (k == "--zmq-bind") { a.zmq_bind = true; a.zmq_bind_set = true; }
    else if (k == "--zmq-connect") { a.zmq_bind = false; a.zmq_bind_set = true; }
    else if (k == "--redis-host") a.redis_host_override = need_arg(i, argc, argv, k);
    else if (k == "--redis-port") a.redis_port = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--redis-key") a.redis_key_override = need_arg(i, argc, argv, k);
    else if (k == "--input-timeout-sec") a.input_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--stale-sec") a.stale_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--lowstate-timeout-sec") a.lowstate_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--activation-blend-sec") a.activation_blend_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--stale-behavior") a.stale_behavior_override = lower(need_arg(i, argc, argv, k));
    else if (k == "--event-log") a.event_log = need_arg(i, argc, argv, k);
    else if (k == "--ep" || k == "--execution-provider") a.execution_provider_override = lower(need_arg(i, argc, argv, k));
    else if (k == "--gpu-device-id" || k == "--device-id") a.gpu_device_id = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--trt-cache-dir") a.trt_cache_dir_override = need_arg(i, argc, argv, k);
    else if (k == "--trt-workspace-size") a.trt_workspace_size = std::stoull(need_arg(i, argc, argv, k));
    else if (k == "--trt-fp16") { a.trt_fp16 = true; a.trt_fp16_set = true; }
    else if (k == "--no-trt-fp16") { a.trt_fp16 = false; a.trt_fp16_set = true; }
    else if (k == "--warmup-iters") a.warmup_iters = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--action-scale") a.action_scale_override = std::stof(need_arg(i, argc, argv, k));
    else if (k == "--ready-file") a.ready_file = need_arg(i, argc, argv, k);
    else if (k == "--start-file") a.start_file = need_arg(i, argc, argv, k);
    else if (k == "--handoff-complete-file") a.handoff_complete_file = need_arg(i, argc, argv, k);
    else if (k == "--handoff-release-file") a.handoff_release_file = need_arg(i, argc, argv, k);
    else if (k == "--preview-file") a.preview_file = need_arg(i, argc, argv, k);
    else if (k == "--commit-file") a.commit_file = need_arg(i, argc, argv, k);
    else if (k == "--discard-file") a.discard_file = need_arg(i, argc, argv, k);
    else if (k == "--preview-timeout-sec") a.preview_timeout_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--cmd-topic" || k == "--candidate-topic") a.cmd_topic = need_arg(i, argc, argv, k);
    else if (k == "--state-topic") a.state_topic = need_arg(i, argc, argv, k);
    else if (k == "--hz") a.inference_hz = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--publish-hz") a.publish_hz = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--sim-lowstate-decimation") a.sim_lowstate_decimation = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--onnx-threads") a.onnx_threads = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--max-frames" || k == "--max-time-step" || k == "--motion-end-step") a.max_frames = std::stoi(need_arg(i, argc, argv, k));
    else if (k == "--duration-sec") a.duration_sec = std::stod(need_arg(i, argc, argv, k));
    else if (k == "--dry-run") a.dry_run = true;
    else if (k == "--dry-run-latency") a.dry_run_latency = true;
    else if (k == "--check-only") a.check_only = true;
    else if (k == "--damping-on-stop") a.damping_on_stop = true;
    else if (k == "--no-damping-on-stop") a.damping_on_stop = false;
    else if (k == "--experimental-no-prestart-hold") a.experimental_no_prestart_hold = true;
    else if (k == "--offline-dry-run-state") a.offline_dry_run_state = true;
    else if (k == "--help") {
      std::cout << "Usage: " << player_label()
                << " --spec actions/.../config/generic_*.yaml [--net enP8p1s0]\n"
                << "  Templates: " << format_supported_templates(player_kind()) << ".\n"
                << "  Variants: " << format_supported_variants(player_kind()) << ".\n";
      if (is_motion_tracking_player()) {
        std::cout << "  TWIST realtime --stale-behavior: hold_then_stop, stop, or default.\n"
                  << "  Humanoid-GPT supports --input offline-npz or realtime HMCP UDP qpos36.\n";
      }
      std::cout
                << "  --check-only loads spec/params/ONNX and runs a zero-input IO check without DDS.\n"
                << "  --dry-run subscribes LowState and runs inference, but does not publish LowCmd.\n"
                << "  --dry-run-latency measures AllMimic/SuperMimic/mjlab_combat receive-to-command latency; requires --dry-run --event-log.\n"
                << "  --lowstate-timeout-sec SEC overrides the spec safety timeout.\n"
                << "  --activation-blend-sec SEC blends q/kp/kd from the final hold command.\n"
                << "  --sim-lowstate-decimation N advances embedded mimic after N fresh simulation states.\n"
                << "  --event-log PATH writes JSONL runtime evidence events.\n"
                << "  --candidate-topic TOPIC publishes private candidate commands.\n"
                << "  --preview-file PATH --commit-file PATH [--discard-file PATH]\n"
                << "    enables transactional local first-command preview for Pilot policies.\n"
                << "  --experimental-no-prestart-hold is an unsafe handoff-ablation baseline.\n";
      std::cout << "  --offline-dry-run-state seeds a synthetic state and requires --dry-run.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }
  if (a.spec.empty()) throw std::runtime_error("--spec is required");
  return a;
}

inline StaleBehavior parse_stale_behavior_value(const std::string& value) {
  const std::string v = lower(value);
  if (v == "default" || v == "stale_to_default") return StaleBehavior::kDefault;
  if (v == "hold_then_stop" || v == "hold-then-stop" || v == "hold") return StaleBehavior::kHoldThenStop;
  if (v == "stop" || v == "fail") return StaleBehavior::kStop;
  throw std::runtime_error("motion_command.stale_behavior must be default, hold_then_stop, or stop");
}

class ProtoMotionCache {
 public:
  ProtoMotionCache() = default;

  explicit ProtoMotionCache(const fs::path& path, int dof_dim = kUnitreeG1ProtocolMotorCount) { load(path, dof_dim); }

  void load(const fs::path& path, int dof_dim = kUnitreeG1ProtocolMotorCount) {
    dof_dim_ = dof_dim;
    auto entries = read_npz_entries(path);
    auto meta_dp = npy_meta(entries, "dof_pos");
    auto meta_br = npy_meta(entries, "body_rot");
    if (meta_dp.shape.size() != 2 || meta_dp.shape[1] != dof_dim_) {
      throw std::runtime_error("ProtoMotions motion dof_pos shape must be [frames," +
                               std::to_string(dof_dim_) + "]");
    }
    if (meta_br.shape.size() != 3 || meta_br.shape[2] != 4) {
      throw std::runtime_error("ProtoMotions motion body_rot shape must be [frames,bodies,4]");
    }
    num_frames = static_cast<int>(meta_dp.shape[0]);
    num_bodies = static_cast<int>(meta_br.shape[1]);
    dof_pos = npy_to_float_vector(entries.at("dof_pos"), "dof_pos");
    dof_vel = npy_to_float_vector(entries.at("dof_vel"), "dof_vel");
    body_rot = npy_to_float_vector(entries.at("body_rot"), "body_rot");
    control_dt = npy_to_float_scalar(entries.at("control_dt"), "control_dt");
    if (entries.count("num_frames")) {
      num_frames = static_cast<int>(npy_to_int64_scalar(entries.at("num_frames"), "num_frames"));
    }
    if (static_cast<int>(dof_pos.size()) != num_frames * dof_dim_ ||
        static_cast<int>(dof_vel.size()) != num_frames * dof_dim_ ||
        static_cast<int>(body_rot.size()) != num_frames * num_bodies * 4) {
      throw std::runtime_error("ProtoMotions motion array size mismatch");
    }
  }

  void dof_at(int frame, std::vector<float>& pos, std::vector<float>& vel) const {
    const int f = std::clamp(frame, 0, num_frames - 1);
    pos.assign(dof_pos.begin() + f * dof_dim_, dof_pos.begin() + (f + 1) * dof_dim_);
    vel.assign(dof_vel.begin() + f * dof_dim_, dof_vel.begin() + (f + 1) * dof_dim_);
  }

  std::array<float, 4> body_rot_at(int frame, int body) const {
    const int f = std::clamp(frame, 0, num_frames - 1);
    const int b = std::clamp(body, 0, num_bodies - 1);
    const size_t off = (static_cast<size_t>(f) * num_bodies + b) * 4;
    return {body_rot[off], body_rot[off + 1], body_rot[off + 2], body_rot[off + 3]};
  }

  int num_frames{0};
  int num_bodies{0};
  int dof_dim_{kUnitreeG1ProtocolMotorCount};
  float control_dt{0.02f};
  std::vector<float> dof_pos;
  std::vector<float> dof_vel;
  std::vector<float> body_rot;
};

struct HumanoidGptReferenceFrame {
  std::array<float, kHumanoidGptProtocolQposDim> qpos{};
  std::array<float, kHumanoidGptProtocolQvelDim> qvel{};
  std::array<float, kHumanoidGptProtocolKeypointCount * 16> kpt2gv_pose{};
  std::array<float, kHumanoidGptProtocolKeypointCount * 6> kpt_cvel_in_gv{};
  std::array<float, 16> gv2wrd_pose = identity_pose4();
  std::array<float, 3> gv_vel{0.f, 0.f, 0.f};
};

class HumanoidGptReferenceCache {
 public:
  void load(const fs::path& path) {
    const auto entries = read_npz_entries(path);
    const std::string k_qpos = first_npz_key(entries, {"qpos"}, "Humanoid-GPT qpos");
    const std::string k_qvel = first_npz_key(entries, {"qvel"}, "Humanoid-GPT qvel");
    const std::string k_pose = first_npz_key(entries, {"kpt2gv_pose", "kpt_npose"}, "Humanoid-GPT keypoint pose");
    const std::string k_cvel = first_npz_key(entries, {"kpt_cvel_in_gv", "kpt_cvel"}, "Humanoid-GPT keypoint cvel");
    const std::string k_gvpose = first_npz_key(entries, {"gv2wrd_pose", "navi_pose"}, "Humanoid-GPT gravity-view pose");
    const std::string k_gvvel = first_npz_key(entries, {"gv_vel", "navi_vel"}, "Humanoid-GPT gravity-view velocity");

    const auto qpos_meta = npy_meta(entries, k_qpos);
    const auto qvel_meta = npy_meta(entries, k_qvel);
    const auto pose_meta = npy_meta(entries, k_pose);
    const auto cvel_meta = npy_meta(entries, k_cvel);
    const auto gvpose_meta = npy_meta(entries, k_gvpose);
    const auto gvvel_meta = npy_meta(entries, k_gvvel);
    if (qpos_meta.shape.size() != 2 || qpos_meta.shape[1] != kHumanoidGptProtocolQposDim) {
      throw std::runtime_error("Humanoid-GPT reference qpos shape must be [frames,36]");
    }
    if (qvel_meta.shape.size() != 2 || qvel_meta.shape[1] != kHumanoidGptProtocolQvelDim) {
      throw std::runtime_error("Humanoid-GPT reference qvel shape must be [frames,35]");
    }
    if (pose_meta.shape.size() != 4 || pose_meta.shape[1] != kHumanoidGptProtocolKeypointCount ||
        pose_meta.shape[2] != 4 || pose_meta.shape[3] != 4) {
      throw std::runtime_error("Humanoid-GPT keypoint pose shape must be [frames,14,4,4]");
    }
    if (cvel_meta.shape.size() != 3 || cvel_meta.shape[1] != kHumanoidGptProtocolKeypointCount || cvel_meta.shape[2] != 6) {
      throw std::runtime_error("Humanoid-GPT keypoint cvel shape must be [frames,14,6]");
    }
    if (gvpose_meta.shape.size() != 3 || gvpose_meta.shape[1] != 4 || gvpose_meta.shape[2] != 4) {
      throw std::runtime_error("Humanoid-GPT gravity-view pose shape must be [frames,4,4]");
    }
    if (gvvel_meta.shape.size() != 2 || gvvel_meta.shape[1] != 3) {
      throw std::runtime_error("Humanoid-GPT gravity-view velocity shape must be [frames,3]");
    }
    num_frames_ = static_cast<int>(qpos_meta.shape[0]);
    if (static_cast<int>(qvel_meta.shape[0]) != num_frames_ ||
        static_cast<int>(pose_meta.shape[0]) != num_frames_ ||
        static_cast<int>(cvel_meta.shape[0]) != num_frames_ ||
        static_cast<int>(gvpose_meta.shape[0]) != num_frames_ ||
        static_cast<int>(gvvel_meta.shape[0]) != num_frames_) {
      throw std::runtime_error("Humanoid-GPT reference arrays must have the same frame count");
    }
    qpos_ = npy_to_float_vector(entries.at(k_qpos), k_qpos);
    qvel_ = npy_to_float_vector(entries.at(k_qvel), k_qvel);
    kpt2gv_pose_ = npy_to_float_vector(entries.at(k_pose), k_pose);
    kpt_cvel_in_gv_ = npy_to_float_vector(entries.at(k_cvel), k_cvel);
    gv2wrd_pose_ = npy_to_float_vector(entries.at(k_gvpose), k_gvpose);
    gv_vel_ = npy_to_float_vector(entries.at(k_gvvel), k_gvvel);
    if (entries.count("frequency")) frequency_hz_ = npy_to_float_scalar(entries.at("frequency"), "frequency");
    if (!(frequency_hz_ > 0.f) || !std::isfinite(frequency_hz_)) frequency_hz_ = 50.f;
  }

  int num_frames() const { return num_frames_; }
  float frequency_hz() const { return frequency_hz_; }
  double duration_sec() const { return num_frames_ > 0 ? static_cast<double>(num_frames_) / std::max(1.f, frequency_hz_) : 0.0; }

  HumanoidGptReferenceFrame frame_at(int frame) const {
    if (num_frames_ <= 0) throw std::runtime_error("Humanoid-GPT reference cache is empty");
    const int f = std::clamp(frame, 0, num_frames_ - 1);
    HumanoidGptReferenceFrame out;
    std::copy_n(qpos_.begin() + static_cast<size_t>(f) * kHumanoidGptProtocolQposDim, kHumanoidGptProtocolQposDim, out.qpos.begin());
    std::copy_n(qvel_.begin() + static_cast<size_t>(f) * kHumanoidGptProtocolQvelDim, kHumanoidGptProtocolQvelDim, out.qvel.begin());
    std::copy_n(kpt2gv_pose_.begin() + static_cast<size_t>(f) * kHumanoidGptProtocolKeypointCount * 16,
                kHumanoidGptProtocolKeypointCount * 16, out.kpt2gv_pose.begin());
    std::copy_n(kpt_cvel_in_gv_.begin() + static_cast<size_t>(f) * kHumanoidGptProtocolKeypointCount * 6,
                kHumanoidGptProtocolKeypointCount * 6, out.kpt_cvel_in_gv.begin());
    std::copy_n(gv2wrd_pose_.begin() + static_cast<size_t>(f) * 16, 16, out.gv2wrd_pose.begin());
    std::copy_n(gv_vel_.begin() + static_cast<size_t>(f) * 3, 3, out.gv_vel.begin());
    return out;
  }

 private:
  int num_frames_{0};
  float frequency_hz_{50.f};
  std::vector<float> qpos_, qvel_, kpt2gv_pose_, kpt_cvel_in_gv_, gv2wrd_pose_, gv_vel_;
};

class TwistOfflineMotionSource {
 public:
  void set_default_frame(const std::array<float, kTwistProtocolMimicDim>& frame) { default_frame_ = frame; }

  void load_csv(const fs::path& path) {
    frames_.clear();
    dt_ = 1.0 / 50.0;
    std::ifstream in(path);
    if (!in.good()) throw std::runtime_error("failed to open TWIST motion cache: " + path.string());
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') {
        parse_header(line);
        continue;
      }
      auto maybe = parse_twist_mimic_packet(line);
      if (!maybe) {
        throw std::runtime_error("bad TWIST mimic row in " + path.string() + ": expected 33 floats");
      }
      frames_.push_back(*maybe);
    }
    if (frames_.empty()) throw std::runtime_error("TWIST motion cache contains no frames: " + path.string());
  }

  const std::array<float, kTwistProtocolMimicDim>& frame_for_time(double t, bool loop, bool& finished) const {
    finished = false;
    if (frames_.empty()) return default_frame_;
    int64_t idx = static_cast<int64_t>(std::floor(t / std::max(1e-6, dt_)));
    if (loop) {
      idx %= static_cast<int64_t>(frames_.size());
      if (idx < 0) idx += frames_.size();
    } else if (idx >= static_cast<int64_t>(frames_.size())) {
      idx = static_cast<int64_t>(frames_.size()) - 1;
      finished = true;
    }
    return frames_.at(static_cast<size_t>(std::max<int64_t>(0, idx)));
  }

  double duration_sec() const { return frames_.empty() ? 0.0 : frames_.size() * dt_; }
  size_t size() const { return frames_.size(); }
  double dt() const { return dt_; }

 private:
  void parse_header(const std::string& line) {
    const std::string key = "dt=";
    const size_t p = line.find(key);
    if (p == std::string::npos) return;
    char* end = nullptr;
    const double v = std::strtod(line.c_str() + p + key.size(), &end);
    if (std::isfinite(v) && v > 0.0) dt_ = v;
  }

  std::vector<std::array<float, kTwistProtocolMimicDim>> frames_;
  std::array<float, kTwistProtocolMimicDim> default_frame_{};
  double dt_{1.0 / 50.0};
};

class TwistUdpReceiver {
 public:
  ~TwistUdpReceiver() { close_socket(); }

  void open_socket(const std::string& bind_ip, int port) {
    if (!is_local_reference_bind(bind_ip)) {
      throw std::runtime_error("TWIST realtime UDP requires loopback bind until an authenticated reference transport is configured");
    }
    close_socket();
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error("TWIST UDP socket failed errno=" + std::to_string(errno));
    int reuse = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind_ip.empty() || bind_ip == "0.0.0.0" || bind_ip == "*") {
      addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr) != 1) {
      throw std::runtime_error("invalid TWIST UDP bind address: " + bind_ip);
    }
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      const int e = errno;
      close_socket();
      throw std::runtime_error("TWIST UDP bind " + bind_ip + ":" + std::to_string(port) + " failed errno=" + std::to_string(e));
    }
    const int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
    std::cout << player_log_prefix() << " TWIST UDP listening on " << bind_ip << ":" << port << "\n";
  }

  bool poll() {
    if (fd_ < 0) return false;
    bool got_any = false;
    while (true) {
      char buf[8192];
      sockaddr_in src{};
      socklen_t slen = sizeof(src);
      const ssize_t n = ::recvfrom(fd_, buf, sizeof(buf) - 1, 0, reinterpret_cast<sockaddr*>(&src), &slen);
      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        throw std::runtime_error("TWIST UDP recv failed errno=" + std::to_string(errno));
      }
      buf[n] = '\0';
      auto maybe = parse_twist_mimic_packet(std::string(buf, static_cast<size_t>(n)));
      if (maybe) {
        last_ = *maybe;
        last_time_ = std::chrono::steady_clock::now();
        ++sequence_;
        has_last_ = true;
        got_any = true;
      } else {
        ++bad_packets_;
        if (bad_packets_ <= 5 || bad_packets_ % 100 == 0) {
          std::cerr << player_log_prefix() << " dropped TWIST UDP packet: expected mimic_obs[33] or 33 CSV floats\n";
        }
      }
    }
    return got_any;
  }

  bool wait_first(double timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_sec));
    while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
      if (poll() && has_last_) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return has_last_;
  }

  TwistMimicSample current_sample(double stale_sec) {
    poll();
    return twist_sample_from_last(has_last_, last_, last_time_, sequence_, stale_sec);
  }

 private:
  void close_socket() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_{-1};
  bool has_last_{false};
  std::array<float, kTwistProtocolMimicDim> last_{};
  std::chrono::steady_clock::time_point last_time_{std::chrono::steady_clock::now()};
  uint64_t sequence_{0};
  size_t bad_packets_{0};
};

class HumanoidGptUdpReceiver {
 public:
  ~HumanoidGptUdpReceiver() { close_socket(); }

  void open_socket(const std::string& bind_ip, int port) {
    if (!is_local_reference_bind(bind_ip)) {
      throw std::runtime_error("Humanoid-GPT realtime UDP requires loopback bind until an authenticated reference transport is configured");
    }
    close_socket();
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error("Humanoid-GPT UDP socket failed errno=" + std::to_string(errno));
    int reuse = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind_ip.empty() || bind_ip == "0.0.0.0" || bind_ip == "*") {
      addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr) != 1) {
      throw std::runtime_error("invalid Humanoid-GPT UDP bind address: " + bind_ip);
    }
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      const int e = errno;
      close_socket();
      throw std::runtime_error("Humanoid-GPT UDP bind " + bind_ip + ":" + std::to_string(port) +
                               " failed errno=" + std::to_string(e));
    }
    const int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
    std::cout << player_log_prefix() << " Humanoid-GPT HMCP UDP listening on " << bind_ip << ":" << port << "\n";
  }

  bool poll() {
    if (fd_ < 0) return false;
    bool got_any = false;
    while (true) {
      std::array<uint8_t, 2048> buf{};
      sockaddr_in src{};
      socklen_t slen = sizeof(src);
      const ssize_t n = ::recvfrom(fd_, buf.data(), buf.size(), 0, reinterpret_cast<sockaddr*>(&src), &slen);
      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        throw std::runtime_error("Humanoid-GPT UDP recv failed errno=" + std::to_string(errno));
      }
      auto maybe = parse_humanoid_gpt_hmcp_packet(buf.data(), static_cast<size_t>(n));
      if (!maybe) {
        maybe = parse_humanoid_gpt_qpos_text_packet(
            std::string(reinterpret_cast<const char*>(buf.data()), static_cast<size_t>(n)));
      }
      if (maybe) {
        last_ = *maybe;
        last_.freshness = SampleFreshness::kFresh;
        last_time_ = std::chrono::steady_clock::now();
        has_last_ = true;
        got_any = true;
      } else {
        ++bad_packets_;
        if (bad_packets_ <= 5 || bad_packets_ % 100 == 0) {
          std::cerr << player_log_prefix() << " dropped Humanoid-GPT UDP packet: expected HMCP qpos[36] or 36 CSV floats\n";
        }
      }
    }
    return got_any;
  }

  bool wait_first(double timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_sec));
    while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
      if (poll() && has_last_) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return has_last_;
  }

  HumanoidGptQposSample current_sample(double stale_sec) {
    poll();
    HumanoidGptQposSample sample = last_;
    if (!has_last_) {
      sample.freshness = SampleFreshness::kMissing;
      sample.age_sec = std::numeric_limits<double>::infinity();
      return sample;
    }
    sample.age_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_time_).count();
    sample.freshness = (stale_sec >= 0.0 && sample.age_sec > stale_sec) ? SampleFreshness::kStale
                                                                         : SampleFreshness::kFresh;
    return sample;
  }

 private:
  void close_socket() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_{-1};
  bool has_last_{false};
  HumanoidGptQposSample last_;
  std::chrono::steady_clock::time_point last_time_{std::chrono::steady_clock::now()};
  size_t bad_packets_{0};
};

class HumanoidGptRefConverter {
 public:
  ~HumanoidGptRefConverter() { close(); }

  void load(const fs::path& xml_path,
            const std::vector<std::string>& joint_names,
            const std::vector<std::string>& keypoint_names,
            float dt) {
    close();
    dt_ = std::max(1e-6f, dt);
    yamlc::require_size("robot.joint_order", kUnitreeG1ProtocolMotorCount, joint_names.size());
    yamlc::require_size("reference.keypoint_order", kHumanoidGptProtocolKeypointCount, keypoint_names.size());
#ifdef ACTION_GO_HAS_MUJOCO
    char error[1024] = {};
    model_ = mj_loadXML(xml_path.string().c_str(), nullptr, error, sizeof(error));
    if (!model_) {
      throw std::runtime_error("MuJoCo failed to load Humanoid-GPT FK XML " + xml_path.string() +
                               ": " + std::string(error));
    }
    data_ = mj_makeData(model_);
    if (!data_) {
      close();
      throw std::runtime_error("MuJoCo failed to allocate Humanoid-GPT FK data");
    }
    if (model_->nq < kHumanoidGptProtocolQposDim || model_->nv < kHumanoidGptProtocolQvelDim) {
      close();
      throw std::runtime_error("Humanoid-GPT FK XML must expose at least nq=36,nv=35");
    }
    for (int i = 0; i < kHumanoidGptProtocolKeypointCount; ++i) {
      const int id = mj_name2id(model_, mjOBJ_BODY, keypoint_names[static_cast<size_t>(i)].c_str());
      if (id < 0) {
        close();
        throw std::runtime_error("Humanoid-GPT FK XML missing body: " + keypoint_names[static_cast<size_t>(i)]);
      }
      body_ids_[i] = id;
    }
    for (int i = 0; i < kUnitreeG1ProtocolMotorCount; ++i) {
      const int id = mj_name2id(model_, mjOBJ_JOINT, joint_names[static_cast<size_t>(i)].c_str());
      if (id < 0) {
        close();
        throw std::runtime_error("Humanoid-GPT FK XML missing joint: " + joint_names[static_cast<size_t>(i)]);
      }
      qpos_ids_[i] = model_->jnt_qposadr[id];
      qvel_ids_[i] = model_->jnt_dofadr[id];
    }
    loaded_ = true;
#else
    (void)xml_path;
    throw std::runtime_error(std::string(player_label()) +
                             " was built without MuJoCo; realtime Humanoid-GPT qpos36 FK is unavailable");
#endif
  }

  bool loaded() const { return loaded_; }

  void reset() {
    have_prev_ = false;
    have_ref_initial_ = false;
    have_robot_initial_ = false;
    prev_qpos_.fill(0.f);
    prev_kpt2wrd_pose_.fill(0.f);
    prev_gv2wrd_pose_ = identity_pose4();
  }

  void set_robot_initial_pose(const std::array<float, 4>& robot_quat, const std::array<float, 2>& robot_xy) {
    robot_initial_yaw_ = quat_yaw(robot_quat);
    robot_initial_xy_ = robot_xy;
    have_robot_initial_ = true;
  }

  HumanoidGptReferenceFrame convert(const std::array<float, kHumanoidGptProtocolQposDim>& raw_qpos) {
    if (!loaded_) {
      throw std::runtime_error("Humanoid-GPT realtime reference converter is not loaded");
    }
    auto qpos = rebias_qpos(raw_qpos);
    HumanoidGptReferenceFrame out;
    out.qpos = qpos;
#ifdef ACTION_GO_HAS_MUJOCO
    for (int i = 0; i < model_->nq; ++i) data_->qpos[i] = 0.0;
    for (int i = 0; i < model_->nv; ++i) data_->qvel[i] = 0.0;
    for (int i = 0; i < 7; ++i) data_->qpos[i] = static_cast<mjtNum>(qpos[i]);
    for (int i = 0; i < kUnitreeG1ProtocolMotorCount; ++i) {
      data_->qpos[qpos_ids_[i]] = static_cast<mjtNum>(qpos[7 + i]);
      if (have_prev_) {
        const float v = (qpos[7 + i] - prev_qpos_[7 + i]) / dt_;
        data_->qvel[qvel_ids_[i]] = static_cast<mjtNum>(v);
        out.qvel[6 + i] = v;
      }
    }
    if (have_prev_) {
      out.qvel[0] = (qpos[0] - prev_qpos_[0]) / dt_;
      out.qvel[1] = (qpos[1] - prev_qpos_[1]) / dt_;
      out.qvel[2] = (qpos[2] - prev_qpos_[2]) / dt_;
      out.qvel[5] = wrap_pi(quat_yaw({qpos[3], qpos[4], qpos[5], qpos[6]}) -
                            quat_yaw({prev_qpos_[3], prev_qpos_[4], prev_qpos_[5], prev_qpos_[6]})) / dt_;
    }
    mj_forward(model_, data_);

    out.gv2wrd_pose = navigation_pose_from_qpos(qpos);
    const auto wrd2gv = pose_inverse(out.gv2wrd_pose);

    std::array<float, kHumanoidGptProtocolKeypointCount * 16> kpt2wrd_pose{};
    for (int k = 0; k < kHumanoidGptProtocolKeypointCount; ++k) {
      std::array<float, 16> T = identity_pose4();
      const int body = body_ids_[k];
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T[r * 4 + c] = static_cast<float>(data_->xmat[body * 9 + r * 3 + c]);
        T[r * 4 + 3] = static_cast<float>(data_->xpos[body * 3 + r]);
      }
      std::copy(T.begin(), T.end(), kpt2wrd_pose.begin() + k * 16);
      const auto T_gv = pose_mul(wrd2gv, T);
      std::copy(T_gv.begin(), T_gv.end(), out.kpt2gv_pose.begin() + k * 16);
      if (have_prev_) {
        std::array<float, 16> T_prev{};
        std::copy_n(prev_kpt2wrd_pose_.begin() + k * 16, 16, T_prev.begin());
        const auto w_wrd = rot_log_from_delta(T_prev, T);
        const std::array<float, 3> v_wrd{
            (T[3] - T_prev[3]) / dt_,
            (T[7] - T_prev[7]) / dt_,
            (T[11] - T_prev[11]) / dt_,
        };
        const auto w_gv = rot_apply(wrd2gv, w_wrd);
        const auto v_gv = rot_apply(wrd2gv, v_wrd);
        out.kpt_cvel_in_gv[k * 6 + 0] = w_gv[0];
        out.kpt_cvel_in_gv[k * 6 + 1] = w_gv[1];
        out.kpt_cvel_in_gv[k * 6 + 2] = w_gv[2];
        out.kpt_cvel_in_gv[k * 6 + 3] = v_gv[0];
        out.kpt_cvel_in_gv[k * 6 + 4] = v_gv[1];
        out.kpt_cvel_in_gv[k * 6 + 5] = v_gv[2];
      }
    }

    if (have_prev_) {
      const float pc = prev_gv2wrd_pose_[0];
      const float ps = prev_gv2wrd_pose_[4];
      const float dx = out.gv2wrd_pose[3] - prev_gv2wrd_pose_[3];
      const float dy = out.gv2wrd_pose[7] - prev_gv2wrd_pose_[7];
      out.gv_vel[0] = (pc * dx + ps * dy) / dt_;
      out.gv_vel[1] = (-ps * dx + pc * dy) / dt_;
      const float cx = out.gv2wrd_pose[0];
      const float sx = out.gv2wrd_pose[4];
      out.gv_vel[2] = std::atan2(pc * sx - ps * cx, pc * cx + ps * sx) / dt_;
    }
    prev_qpos_ = qpos;
    prev_kpt2wrd_pose_ = kpt2wrd_pose;
    prev_gv2wrd_pose_ = out.gv2wrd_pose;
    have_prev_ = true;
#endif
    return out;
  }

 private:
  void close() {
#ifdef ACTION_GO_HAS_MUJOCO
    if (data_) {
      mj_deleteData(data_);
      data_ = nullptr;
    }
    if (model_) {
      mj_deleteModel(model_);
      model_ = nullptr;
    }
#endif
    loaded_ = false;
  }

  std::array<float, kHumanoidGptProtocolQposDim> rebias_qpos(const std::array<float, kHumanoidGptProtocolQposDim>& raw) {
    if (!have_robot_initial_) return raw;
    auto qpos = raw;
    const std::array<float, 4> ref_quat{qpos[3], qpos[4], qpos[5], qpos[6]};
    if (!have_ref_initial_) {
      ref_initial_yaw_ = quat_yaw(ref_quat);
      ref_initial_xy_ = {qpos[0], qpos[1]};
      have_ref_initial_ = true;
    }
    const float ref_yaw = quat_yaw(ref_quat);
    const float d_yaw = wrap_pi(ref_yaw - ref_initial_yaw_);
    const std::array<float, 2> d_xy{qpos[0] - ref_initial_xy_[0], qpos[1] - ref_initial_xy_[1]};
    const float yaw_offset = wrap_pi(robot_initial_yaw_ - ref_initial_yaw_);
    const float c = std::cos(yaw_offset);
    const float s = std::sin(yaw_offset);
    qpos[0] = robot_initial_xy_[0] + c * d_xy[0] - s * d_xy[1];
    qpos[1] = robot_initial_xy_[1] + s * d_xy[0] + c * d_xy[1];
    const float new_yaw = robot_initial_yaw_ + d_yaw;
    const float apply_yaw = wrap_pi(new_yaw - ref_yaw);
    const auto q_new = quat_mul(axis_quat(apply_yaw, 'z'), ref_quat);
    qpos[3] = q_new[0];
    qpos[4] = q_new[1];
    qpos[5] = q_new[2];
    qpos[6] = q_new[3];
    return qpos;
  }

  float dt_{0.02f};
  bool loaded_{false};
  bool have_prev_{false};
  bool have_robot_initial_{false};
  bool have_ref_initial_{false};
  float robot_initial_yaw_{0.f};
  float ref_initial_yaw_{0.f};
  std::array<float, 2> robot_initial_xy_{0.f, 0.f};
  std::array<float, 2> ref_initial_xy_{0.f, 0.f};
  std::array<int, kHumanoidGptProtocolKeypointCount> body_ids_{};
  std::array<int, kUnitreeG1ProtocolMotorCount> qpos_ids_{};
  std::array<int, kUnitreeG1ProtocolMotorCount> qvel_ids_{};
  std::array<float, kHumanoidGptProtocolQposDim> prev_qpos_{};
  std::array<float, kHumanoidGptProtocolKeypointCount * 16> prev_kpt2wrd_pose_{};
  std::array<float, 16> prev_gv2wrd_pose_ = identity_pose4();
#ifdef ACTION_GO_HAS_MUJOCO
  mjModel* model_{nullptr};
  mjData* data_{nullptr};
#endif
};

class TwistZmqReceiver {
 public:
  ~TwistZmqReceiver() { close_socket(); }

  void open_socket(const std::string& endpoint, const std::string& topic, bool bind_socket) {
#ifdef ACTION_GO_HAS_ZMQ
    if (!is_local_reference_endpoint(endpoint)) {
      throw std::runtime_error("TWIST ZMQ reference endpoint must be local until an authenticated reference transport is configured");
    }
    close_socket();
    ctx_ = zmq_ctx_new();
    if (!ctx_) throw std::runtime_error("TWIST ZMQ context creation failed");
    sock_ = zmq_socket(ctx_, ZMQ_SUB);
    if (!sock_) throw std::runtime_error("TWIST ZMQ SUB socket creation failed");
    int linger = 0;
    zmq_setsockopt(sock_, ZMQ_LINGER, &linger, sizeof(linger));
    int conflate = 1;
    zmq_setsockopt(sock_, ZMQ_CONFLATE, &conflate, sizeof(conflate));
    if (zmq_setsockopt(sock_, ZMQ_SUBSCRIBE, topic.data(), topic.size()) != 0) {
      throw std::runtime_error("TWIST ZMQ subscribe failed");
    }
    const int rc = bind_socket ? zmq_bind(sock_, endpoint.c_str()) : zmq_connect(sock_, endpoint.c_str());
    if (rc != 0) {
      throw std::runtime_error(std::string("TWIST ZMQ ") + (bind_socket ? "bind" : "connect") +
                               " failed endpoint=" + endpoint + " errno=" + std::to_string(errno));
    }
    std::cout << player_log_prefix() << " TWIST ZMQ SUB " << (bind_socket ? "bind" : "connect")
              << " endpoint=" << endpoint << " topic='" << topic << "'\n";
#else
    (void)endpoint; (void)topic; (void)bind_socket;
    throw std::runtime_error(std::string(player_label()) +
                             " was built without libzmq; install libzmq and rebuild for TWIST ZMQ");
#endif
  }

  bool poll() {
#ifdef ACTION_GO_HAS_ZMQ
    if (!sock_) return false;
    bool got_any = false;
    while (true) {
      std::string joined;
      std::string last_part;
      bool have_msg = false;
      while (true) {
        char buf[8192];
        const int n = zmq_recv(sock_, buf, sizeof(buf), ZMQ_DONTWAIT);
        if (n < 0) {
          if (errno == EAGAIN) return got_any;
          throw std::runtime_error("TWIST ZMQ recv failed errno=" + std::to_string(errno));
        }
        have_msg = true;
        std::string part(buf, buf + n);
        if (!joined.empty()) joined.push_back(' ');
        joined += part;
        last_part = std::move(part);
        int more = 0;
        size_t more_size = sizeof(more);
        zmq_getsockopt(sock_, ZMQ_RCVMORE, &more, &more_size);
        if (!more) break;
      }
      if (!have_msg) break;
      auto maybe = parse_twist_mimic_packet(last_part);
      if (!maybe && joined != last_part) maybe = parse_twist_mimic_packet(joined);
      if (maybe) {
        last_ = *maybe;
        last_time_ = std::chrono::steady_clock::now();
        ++sequence_;
        has_last_ = true;
        got_any = true;
      } else {
        ++bad_packets_;
        if (bad_packets_ <= 5 || bad_packets_ % 100 == 0) {
          std::cerr << player_log_prefix() << " dropped TWIST ZMQ packet: expected mimic_obs[33] or 33 CSV floats\n";
        }
      }
    }
    return got_any;
#else
    return false;
#endif
  }

  bool wait_first(double timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_sec));
    while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
      if (poll() && has_last_) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return has_last_;
  }

  TwistMimicSample current_sample(double stale_sec) {
    poll();
    return twist_sample_from_last(has_last_, last_, last_time_, sequence_, stale_sec);
  }

 private:
  void close_socket() {
#ifdef ACTION_GO_HAS_ZMQ
    if (sock_) {
      zmq_close(sock_);
      sock_ = nullptr;
    }
    if (ctx_) {
      zmq_ctx_term(ctx_);
      ctx_ = nullptr;
    }
#endif
  }

#ifdef ACTION_GO_HAS_ZMQ
  void* ctx_{nullptr};
  void* sock_{nullptr};
#endif
  bool has_last_{false};
  std::array<float, kTwistProtocolMimicDim> last_{};
  std::chrono::steady_clock::time_point last_time_{std::chrono::steady_clock::now()};
  uint64_t sequence_{0};
  size_t bad_packets_{0};
};

class SonicZmqReceiver {
 public:
  ~SonicZmqReceiver() { close_socket(); }

  void open_socket(const std::string& endpoint, const std::string& topic, bool bind_socket) {
#ifdef ACTION_GO_HAS_ZMQ
    if (!is_local_reference_endpoint(endpoint)) {
      throw std::runtime_error("SONIC ZMQ reference endpoint must be local until an authenticated reference transport is configured");
    }
    close_socket();
    pose_topic_ = topic;
    ctx_ = zmq_ctx_new();
    if (!ctx_) throw std::runtime_error("SONIC ZMQ context creation failed");
    sock_ = zmq_socket(ctx_, ZMQ_SUB);
    if (!sock_) throw std::runtime_error("SONIC ZMQ SUB socket creation failed");
    int linger = 0;
    zmq_setsockopt(sock_, ZMQ_LINGER, &linger, sizeof(linger));
    for (const std::string& subscribed : {pose_topic_, std::string("planner"), std::string("command")}) {
      if (zmq_setsockopt(sock_, ZMQ_SUBSCRIBE, subscribed.data(), subscribed.size()) != 0) {
        throw std::runtime_error("SONIC ZMQ subscribe failed for topic=" + subscribed);
      }
    }
    const int rc = bind_socket ? zmq_bind(sock_, endpoint.c_str()) : zmq_connect(sock_, endpoint.c_str());
    if (rc != 0) {
      throw std::runtime_error(std::string("SONIC ZMQ ") + (bind_socket ? "bind" : "connect") +
                               " failed endpoint=" + endpoint + " errno=" + std::to_string(errno));
    }
    std::cout << player_log_prefix() << " SONIC ZMQ SUB " << (bind_socket ? "bind" : "connect")
              << " endpoint=" << endpoint << " topics='" << pose_topic_ << ",planner,command'\n";
#else
    (void)endpoint; (void)topic; (void)bind_socket;
    throw std::runtime_error(std::string(player_label()) + " was built without libzmq");
#endif
  }

  bool poll() {
#ifdef ACTION_GO_HAS_ZMQ
    if (!sock_) return false;
    bool got_any = false;
    while (true) {
      std::vector<uint8_t> packet(8 * 1024 * 1024);
      const size_t capacity = packet.size();
      const int size = zmq_recv(sock_, packet.data(), capacity, ZMQ_DONTWAIT);
      if (size < 0) {
        if (errno == EAGAIN) return got_any;
        throw std::runtime_error("SONIC ZMQ recv failed errno=" + std::to_string(errno));
      }
      if (static_cast<size_t>(size) > capacity) {
        ++bad_packets_;
        std::cerr << player_log_prefix() << " dropped oversized SONIC ZMQ packet bytes=" << size << "\n";
        continue;
      }
      packet.resize(static_cast<size_t>(size));
      try {
        if (packet.size() >= pose_topic_.size() &&
            std::memcmp(packet.data(), pose_topic_.data(), pose_topic_.size()) == 0) {
          accept_pose(packet);
          got_any = true;
        } else if (packet.size() >= 7 && std::memcmp(packet.data(), "planner", 7) == 0) {
          std::string signature;
          auto command = action_go::runtime::parse_sonic_planner_packet(packet, "planner", &signature);
          if (has_planner_layout_ && signature != planner_layout_) {
            throw std::runtime_error("planner field order, dtype, shape, or endian changed");
          }
          planner_layout_ = std::move(signature);
          has_planner_layout_ = true;
          planner_command_ = command;
          planner_time_ = std::chrono::steady_clock::now();
          ++planner_sequence_;
          has_planner_ = true;
          got_any = true;
        } else if (packet.size() >= 7 && std::memcmp(packet.data(), "command", 7) == 0) {
          std::string signature;
          auto command = action_go::runtime::parse_sonic_command_packet(packet, "command", &signature);
          // delta_heading is optional, so command layouts may legitimately switch between two known forms.
          control_command_ = command;
          control_time_ = std::chrono::steady_clock::now();
          ++control_sequence_;
          has_control_ = true;
          got_any = true;
        } else {
          throw std::runtime_error("unknown SONIC ZMQ topic");
        }
      } catch (const std::exception& e) {
        ++bad_packets_;
        if (bad_packets_ <= 5 || bad_packets_ % 100 == 0) {
          std::cerr << player_log_prefix() << " dropped SONIC ZMQ packet: " << e.what() << "\n";
        }
      }
    }
#else
    return false;
#endif
  }

  bool wait_first(double timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_sec));
    while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
      if (poll() && has_last_) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return has_last_;
  }

  SonicReferenceSample current_sample(double stale_sec) {
    poll();
    SonicReferenceSample sample;
    if (!has_last_) return sample;
    sample.frames = last_frames_;
    sample.sequence = sequence_;
    sample.age_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_time_).count();
    sample.freshness = sample.age_sec > stale_sec ? SampleFreshness::kStale : SampleFreshness::kFresh;
    return sample;
  }

  SonicPlannerSample current_planner_sample(double stale_sec) {
    poll();
    SonicPlannerSample sample;
    if (!has_planner_) return sample;
    sample.command = planner_command_;
    sample.sequence = planner_sequence_;
    sample.age_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - planner_time_).count();
    sample.freshness = sample.age_sec > stale_sec ? SampleFreshness::kStale : SampleFreshness::kFresh;
    return sample;
  }

  SonicControlSample current_control_sample() {
    poll();
    SonicControlSample sample;
    if (!has_control_) return sample;
    sample.present = true;
    sample.command = control_command_;
    sample.sequence = control_sequence_;
    sample.age_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - control_time_).count();
    return sample;
  }

  bool has_pose() const { return has_last_; }
  bool has_planner() const { return has_planner_; }
  bool has_control() const { return has_control_; }

 private:
  void accept_pose(const std::vector<uint8_t>& packet) {
    std::string layout_signature;
    auto frames = action_go::runtime::parse_sonic_zmq_packet(packet, pose_topic_, &layout_signature);
    if (has_layout_ && layout_signature != layout_signature_) {
      throw std::runtime_error("pose field order, dtype, shape, or endian changed during the active stream");
    }
    const bool explicit_index = !frames.empty() && frames.front().has_explicit_frame_index;
    if (has_index_mode_ && explicit_index != explicit_index_mode_) {
      throw std::runtime_error("frame_index presence changed during the active stream");
    }
    if (!explicit_index) {
      const int64_t first = has_last_ ? last_frame_index_ + 1 : 0;
      for (size_t i = 0; i < frames.size(); ++i) frames[i].frame_index = first + static_cast<int64_t>(i);
    }
    if (has_last_ && !frames.empty() && frames.back().frame_index == last_frame_index_) return;
    explicit_index_mode_ = explicit_index;
    has_index_mode_ = true;
    layout_signature_ = std::move(layout_signature);
    has_layout_ = true;
    last_frame_index_ = frames.back().frame_index;
    last_frames_ = std::move(frames);
    last_time_ = std::chrono::steady_clock::now();
    ++sequence_;
    has_last_ = true;
  }

  void close_socket() {
#ifdef ACTION_GO_HAS_ZMQ
    if (sock_) { zmq_close(sock_); sock_ = nullptr; }
    if (ctx_) { zmq_ctx_term(ctx_); ctx_ = nullptr; }
#endif
  }

#ifdef ACTION_GO_HAS_ZMQ
  void* ctx_{nullptr};
  void* sock_{nullptr};
#endif
  std::string pose_topic_;
  bool has_last_{false};
  bool has_index_mode_{false}, explicit_index_mode_{false};
  bool has_layout_{false};
  std::string layout_signature_;
  std::vector<SonicReferenceFrame> last_frames_;
  std::chrono::steady_clock::time_point last_time_{std::chrono::steady_clock::now()};
  uint64_t sequence_{0};
  int64_t last_frame_index_{-1};
  bool has_planner_{false}, has_control_{false}, has_planner_layout_{false};
  SonicPlannerCommand planner_command_{};
  SonicControlCommand control_command_{};
  std::string planner_layout_;
  std::chrono::steady_clock::time_point planner_time_{std::chrono::steady_clock::now()};
  std::chrono::steady_clock::time_point control_time_{std::chrono::steady_clock::now()};
  uint64_t planner_sequence_{0}, control_sequence_{0};
  size_t bad_packets_{0};
};

class TwistRedisReceiver {
 public:
  ~TwistRedisReceiver() { close_socket(); }

  void open_socket(const std::string& host, int port, const std::string& key) {
    host_ = host;
    port_ = port;
    key_ = key;
    close_socket();
    std::cout << player_log_prefix() << " TWIST Redis GET redis://" << host_ << ":" << port_
              << "/ key='" << key_ << "'\n";
  }

  bool poll() {
    if (!ensure_connected()) return false;
    const std::string cmd = "*2\r\n$3\r\nGET\r\n$" + std::to_string(key_.size()) + "\r\n" + key_ + "\r\n";
    if (!send_all(cmd)) {
      close_socket();
      return false;
    }
    std::string value;
    if (!read_bulk_string(value)) {
      close_socket();
      return false;
    }
    if (value.empty()) return false;
    auto maybe = parse_twist_mimic_packet(value);
    if (maybe) {
      last_ = *maybe;
      last_time_ = std::chrono::steady_clock::now();
      ++sequence_;
      has_last_ = true;
      return true;
    }
    ++bad_packets_;
    if (bad_packets_ <= 5 || bad_packets_ % 100 == 0) {
      std::cerr << player_log_prefix() << " dropped TWIST Redis value key='" << key_
                << "': expected mimic_obs[33] or 33 CSV floats\n";
    }
    return false;
  }

  bool wait_first(double timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_sec));
    while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline) {
      if (poll() && has_last_) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return has_last_;
  }

  TwistMimicSample current_sample(double stale_sec) {
    poll();
    return twist_sample_from_last(has_last_, last_, last_time_, sequence_, stale_sec);
  }

 private:
  bool ensure_connected() {
    if (fd_ >= 0) return true;
    const auto now = std::chrono::steady_clock::now();
    if (now < next_reconnect_) return false;
    next_reconnect_ = now + std::chrono::milliseconds(500);

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    const int gai = getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res);
    if (gai != 0) return false;
    for (auto* rp = res; rp; rp = rp->ai_next) {
      int fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (fd < 0) continue;
      struct timeval tv{};
      tv.tv_sec = 0;
      tv.tv_usec = 2000;
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
      if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
        fd_ = fd;
        freeaddrinfo(res);
        return true;
      }
      ::close(fd);
    }
    freeaddrinfo(res);
    return false;
  }

  bool send_all(const std::string& data) {
    const char* p = data.data();
    size_t left = data.size();
    while (left > 0) {
      const ssize_t n = ::send(fd_, p, left, MSG_NOSIGNAL);
      if (n <= 0) return false;
      p += n;
      left -= static_cast<size_t>(n);
    }
    return true;
  }

  bool recv_byte(char& c) {
    const ssize_t n = ::recv(fd_, &c, 1, 0);
    return n == 1;
  }

  bool read_line(std::string& line) {
    line.clear();
    char prev = 0;
    for (int i = 0; i < 4096; ++i) {
      char c = 0;
      if (!recv_byte(c)) return false;
      if (prev == '\r' && c == '\n') {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
      }
      line.push_back(c);
      prev = c;
    }
    return false;
  }

  bool read_exact(std::string& out, size_t n) {
    out.assign(n, '\0');
    size_t got = 0;
    while (got < n) {
      const ssize_t r = ::recv(fd_, out.data() + got, n - got, 0);
      if (r <= 0) return false;
      got += static_cast<size_t>(r);
    }
    return true;
  }

  bool read_bulk_string(std::string& value) {
    value.clear();
    std::string line;
    if (!read_line(line) || line.empty()) return false;
    if (line[0] == '$') {
      const int len = std::stoi(line.substr(1));
      if (len < 0) return true;
      std::string payload;
      if (!read_exact(payload, static_cast<size_t>(len))) return false;
      char cr = 0, lf = 0;
      if (!recv_byte(cr) || !recv_byte(lf)) return false;
      value = std::move(payload);
      return true;
    }
    if (line[0] == '+') {
      value = line.substr(1);
      return true;
    }
    if (line[0] == '-') {
      std::cerr << player_log_prefix() << " TWIST Redis error: " << line << "\n";
      return false;
    }
    return false;
  }

  void close_socket() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_{-1};
  std::string host_{"127.0.0.1"};
  int port_{6379};
  std::string key_{"action_mimic_g1"};
  std::chrono::steady_clock::time_point next_reconnect_{std::chrono::steady_clock::now()};
  bool has_last_{false};
  std::array<float, kTwistProtocolMimicDim> last_{};
  std::chrono::steady_clock::time_point last_time_{std::chrono::steady_clock::now()};
  uint64_t sequence_{0};
  size_t bad_packets_{0};
};

class GenericPolicyPlayer {
 public:
  explicit GenericPolicyPlayer(Args args);
  ~GenericPolicyPlayer();

  void run();

  void log_process_exit(bool ok, const std::string& reason) noexcept;

  void emergency_stop_after_error() noexcept;

 private:
  enum class CommandPhase {
    kHold,
    kPreview,
    kActive,
    kDamping,
  };
  enum class PreviewDecision {
    kDisabled,
    kCommit,
    kDiscard,
    kTimeout,
    kStopped,
  };
  struct RobotState {
    std::array<float, 4> quat{1.f, 0.f, 0.f, 0.f};
    std::array<float, 3> gyro{0.f, 0.f, 0.f};
    std::array<uint8_t, 40> wireless_remote{};
    std::vector<float> q;
    std::vector<float> dq;
    std::chrono::steady_clock::time_point stamp{};
    uint64_t seq{0};
    uint32_t source_tick_ms{0};
    int mode_machine{0};
    bool connected{false};
  };
  struct Command { float vx{0.f}, vy{0.f}, yaw{0.f}; };

  struct BeyondResult {
    std::vector<float> actions;
    std::vector<float> joint_pos;
    std::vector<float> joint_vel;
    std::vector<float> body_quat_w;
  };
  struct HoloResult {
    std::vector<float> actions;
    std::vector<float> joint_pos;
    std::vector<float> joint_vel;
    std::array<float, 4> ref_quat_xyzw{0.f, 0.f, 0.f, 1.f};
  };
  struct JumpStats {
    double all{0.0};
    int all_model{-1};
    int all_motor{-1};
    double lower_body{0.0};
    int lower_body_model{-1};
    int lower_body_motor{-1};
  };
  enum class TwistTargetStatus { kCommand, kHoldLast, kStop };
  enum class HumanoidGptTargetStatus { kCommand, kHoldLast, kStop };
  struct PolicyRuntimeModule {
    const char* variant;
    GenericPlayerKind player_kind;
    void (GenericPolicyPlayer::*load_spec_fields)(const YAML::Node&);
    void (GenericPolicyPlayer::*load_params)();
    void (GenericPolicyPlayer::*post_load)();
    void (GenericPolicyPlayer::*schema_summary)();
    void (GenericPolicyPlayer::*check_only)();
    void (GenericPolicyPlayer::*control_loop)();
  };

  void validate_player_scope() const;

  const PolicyRuntimeModule& runtime_module() const;

  static std::string join_names(const std::vector<std::string>& names);

  static std::string join_ints(const std::vector<int>& values);

  static void require_name_list_equal(const std::string& field,
                                      const std::vector<std::string>& actual,
                                      const std::vector<const char*>& expected_raw);

  void bind_protocol_constraint();

  void validate_spec_against_protocol() const;

  void load_spec_bfm(const YAML::Node& y);

  void load_spec_pbfm(const YAML::Node& y);

  void post_load_pbfm();

  void check_only_pbfm();

  std::vector<float> pbfm_depth_observation();

  std::vector<float> infer_pbfm(const RobotState& state, const BfmObservation& observation);

  void load_spec_unitree(const YAML::Node& y);

  void load_spec_beyondmimic(const YAML::Node& y);

  void load_spec_supermimic(const YAML::Node& y);

  void load_supermimic_params();

  void post_load_supermimic();

  void schema_summary_supermimic();

  void check_only_supermimic();

  void control_loop_supermimic();

  template <typename Controller>
  void check_only_motion_controller(Controller& controller);

  template <typename Controller>
  void control_loop_motion_controller(Controller& controller);

  void load_spec_holosoma(const YAML::Node& y);

  void load_spec_protomotions(const YAML::Node& y);

  void load_spec_twist(const YAML::Node& y);

  void load_spec_humanoid_gpt(const YAML::Node& y);

  void load_spec_sonic(const YAML::Node& y);

  void load_spec();

  void load_variant_params();

  void post_load_variant();

  void post_load_bfm();

  void post_load_unitree();

  void post_load_beyondmimic();

  void post_load_holosoma();

  void post_load_protomotions();

  void post_load_twist();

  void post_load_humanoid_gpt();

  void post_load_sonic();

  void load_bfm_params();

  void load_unitree_params();

  void load_beyond_params();

  void load_holosoma_params();

  void load_protomotions_params();

  void load_twist_params();

  void load_humanoid_gpt_params();

  void load_sonic_params();

  fs::path ensure_twist_motion_cache();

  void setup_twist_input();

  void setup_humanoid_gpt_reference();

  void setup_sonic_reference();

  void validate_joint_index() const;

  void schema_summary_common() const;

  void schema_summary_bfm();

  void schema_summary_unitree();

  void schema_summary_beyondmimic();

  void schema_summary_holosoma();

  void schema_summary_protomotions();

  void schema_summary_twist();

  void schema_summary_humanoid_gpt();

  void schema_summary_sonic();

  void check_only();

  void check_only_bfm();

  void check_only_unitree();

  void check_only_beyondmimic();

  void check_only_holosoma();

  void check_only_protomotions();

  void check_only_twist();

  void check_only_humanoid_gpt();

  void check_only_sonic();

  static void print_names(const std::vector<std::string>& names);

  static int advance_deadline(std::chrono::steady_clock::time_point& next,
                              std::chrono::steady_clock::duration dt,
                              std::chrono::steady_clock::time_point now);

  void on_low_state(const void* message);

  RobotState snapshot() const;

  CommandSafetyState command_safety_state(const RobotState& s) const;

  void wait_for_state();

  void touch_file(const std::string& path);

  bool preview_mode_enabled() const;

  PreviewDecision wait_preview_decision(const std::vector<float>& q,
                                         const std::vector<float>& kp,
                                         const std::vector<float>& kd,
                                         int mode_machine,
                                         int publish_hz,
                                         int warmup_steps);

  void publish(const std::vector<float>& q,
               const std::vector<float>& kp,
               const std::vector<float>& kd,
               int mode_machine,
               bool allow_stale_state = false,
               CommandPhase phase = CommandPhase::kActive,
               CommandTiming timing = {});

  void publish_damping_for_stop(bool allow_stale_state = false);
  void wait_postrun_handoff(int publish_hz);

  void hold_current_pose(const RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                         float hold_kp = 20.f, float hold_kd = 1.f) const;

  void wait_start_file(std::vector<float>& hold_q, const std::vector<float>& hold_kp,
                       const std::vector<float>& hold_kd, int publish_hz,
                       const std::function<void()>& standby_health_check = {},
                       bool follow_current_pose = true);

  std::vector<float> bfm_infer(const std::vector<float>& obs);

  bool build_bfm_target(const RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                        std::vector<float>& action_out);

  void control_loop_single_bfm();

  Command unitree_command_from_remote(const std::array<uint8_t, 40>& remote) const;

  std::vector<float> build_unitree_obs(const RobotState& s, const Command& c);

  bool build_unitree_target(const RobotState& s, std::vector<float>& target, std::vector<float>& kp, std::vector<float>& kd,
                            Command& cmd, std::vector<float>& action);

  void control_loop_single_unitree();

  BeyondResult beyond_inference(const std::vector<float>& obs, float t);

  void reset_beyond_controller();

  bool build_beyond_target(const RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd);

  void control_loop_beyondmimic();

  HoloResult holosoma_inference(const std::vector<float>& obs, float timestep);

  std::vector<float> model_to_motor(const std::vector<float>& model_vec, float fill = 0.f) const;

  std::vector<float> motor_to_model(const std::vector<float>& motor_vec) const;

  std::vector<float> clamp_model_q(const std::vector<float>& q_model) const;

  void reset_holosoma_motion_buffers();

  std::array<float, 4> torso_quat_from_model_q(const std::array<float, 4>& base_quat, const std::vector<float>& q_model) const;

  std::array<float, 4> torso_quat_from_state(const RobotState& s) const;

  std::vector<float> build_holosoma_obs(const RobotState& s) const;

  int holosoma_timestep_from_wall_clock() const;

  JumpStats target_jump_stats(const std::vector<float>& a, const std::vector<float>& b) const;

  std::string joint_desc(int model_i, int motor_i) const;

  bool check_holosoma_target_safety(const RobotState& state, const std::vector<float>& q_target, bool motion_clip_progressing);

  bool build_holosoma_policy_command(const RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                                     bool motion_clip_progressing);

  void holosoma_set_current_pose_hold();

  void holosoma_prepare_startup_pose(int publish_hz);

  void holosoma_capture_yaw_offsets();

  void control_loop_holosoma();

  std::array<float, 4> protomotions_torso_anchor_from_base_and_waist(const std::array<float, 4>& base_wxyz,
                                                                      const std::vector<float>& q) const;

  std::array<float, 4> protomotions_anchor_quat(const RobotState& s) const;

  std::vector<float> protomotions_infer(const RobotState& s);

  void protomotions_advance();

  bool protomotions_done() const;

  void control_loop_protomotions();

  void reset_twist_controller();

  bool wait_first_twist_realtime();

  const char* twist_stale_behavior_name() const;

  TwistMimicSample current_twist_realtime_mimic();

  TwistMimicSample current_twist_mimic(int64_t policy_step, bool& offline_finished);

  void split_twist_mimic(const std::array<float, kTwistProtocolMimicDim>& mimic,
                         std::vector<float>& policy_mimic,
                         std::array<float, 2>& wrist_roll) const;

  std::vector<float> build_twist_model_input(const RobotState& s, const std::array<float, kTwistProtocolMimicDim>& mimic);

  std::vector<float> twist_infer(const std::vector<float>& obs);

  TwistTargetStatus build_twist_target(const RobotState& s, int64_t policy_step,
                                       std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                                       std::vector<float>& raw_action);

  void control_loop_twist();

  HumanoidGptReferenceFrame default_humanoid_gpt_reference() const;

  bool wait_first_humanoid_gpt_realtime();

  const char* humanoid_gpt_stale_behavior_name() const;

  bool humanoid_gpt_offline_done() const;

  HumanoidGptTargetStatus update_humanoid_gpt_reference(const RobotState& s, int64_t policy_step);

  std::vector<float> build_humanoid_gpt_obs(const RobotState& s,
                                            const HumanoidGptReferenceFrame& ref_curr,
                                            const HumanoidGptReferenceFrame& ref_next) const;

  std::vector<float> humanoid_gpt_infer(const std::vector<float>& obs);

  HumanoidGptTargetStatus build_humanoid_gpt_target(const RobotState& s, int64_t policy_step,
                                                    std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                                                    std::vector<float>& raw_action);

  void reset_humanoid_gpt_runtime(const RobotState& s);

  void control_loop_humanoid_gpt();

  bool wait_first_sonic_realtime();

  SonicRobotFrame sonic_robot_frame(const RobotState& state) const;

  bool update_sonic_reference(const RobotState& state, int64_t policy_step);
  void require_sonic_realtime_fresh(const char* phase, bool revoke_ready);
  bool prepare_sonic_pico_source(const RobotState& state, bool require_started);

  bool build_sonic_target(const RobotState& state, int64_t policy_step,
                          std::vector<float>& target, std::vector<float>& raw_action);

  void reset_sonic_runtime(const RobotState& state);

  void control_loop_sonic();

  Args args_;
  const PolicyProtocolConstraint* protocol_{nullptr};
  fs::path spec_path_, model_path_, runtime_model_path_, params_path_, trajectory_path_;
  std::string name_, template_, variant_, latent_file_, bfm_latent_sequence_file_;
  std::string net_{"enP8p1s0"}, cmd_topic_{"rt/user_lowcmd"}, state_topic_{"rt/lowstate"}, ready_file_, start_file_;
  std::string handoff_complete_file_, handoff_release_file_;
  std::string preview_file_, commit_file_, discard_file_;
  std::vector<std::string> expected_input_names_, expected_output_names_;
  std::string obs_input_name_, time_input_name_{"time_step"}, action_output_name_;
  int inference_hz_{50}, publish_hz_{500}, onnx_threads_{2}, mode_machine_{5}, stats_interval_{100};
  int sim_lowstate_decimation_{0};
  int max_frames_{0};
  double duration_sec_{0.0}, target_interp_sec_{0.02}, startup_interp_sec_{0.0}, preview_timeout_sec_{5.0};
  double activation_blend_sec_{0.10};
  bool damping_on_stop_{false};

  int motor_count_{0};
  int configured_obs_dim_{0};
  int configured_action_dim_{0};
  int num_actions_{0}, num_obs_{0};
  std::vector<float> default_q_, action_scale_, kp_, kd_, lower_limits_, upper_limits_, torque_limits_, last_action_;
  std::vector<float> last_published_q_, last_published_kp_, last_published_kd_;
  std::vector<int> joint_index_;
  std::vector<std::string> joint_order_;
  int64_t policy_counter_{0};

  int bfm_actor_obs_dim_{0};
  int bfm_z_dim_{0};
  int bfm_history_steps_{0};
  BfmHistoryTiming bfm_history_timing_{BfmHistoryTiming::kIncludeCurrent};
  int bfm_warmup_steps_{5};
  float bfm_action_rescale_{5.f};
  float bfm_action_rescale_override_{std::numeric_limits<float>::quiet_NaN()};
  bool bfm_latent_sequence_loop_{true};
  size_t bfm_latent_sequence_frames_{0};
  size_t bfm_latent_sequence_index_{0};
  std::vector<float> bfm_latent_sequence_;
  std::vector<float> bfm_z_, bfm_last_action_scaled_;
  BfmObservationHistory bfm_history_;
  PbfmSpec pbfm_spec_;
  std::unique_ptr<BfmContextStream> pbfm_context_stream_;
  float pbfm_reference_start_yaw_{0.f};
  std::unique_ptr<RealSenseDepthStream> pbfm_realsense_;
  PbfmDepthHistory pbfm_depth_history_;
#ifdef ACTION_GO_HAS_ZMQ
  std::unique_ptr<SharedMemoryDepthStream> pbfm_depth_stream_;
#endif
  std::vector<float> pbfm_file_depth_;
  std::vector<float> pbfm_estimated_velocity_;
  std::ofstream pbfm_trace_;
  float pbfm_initial_yaw_{0.f};
  float pbfm_command_yaw_{0.f};
  size_t pbfm_reference_step_{0};
  std::string pbfm_last_motion_state_;
  PbfmHeadingController pbfm_heading_controller_;

  int unitree_base_obs_dim_{0};
  int unitree_obs_dim_{0};
  std::vector<int> unitree_allowed_obs_dims_;
  std::string unitree_command_source_{"remote"};
  float unitree_lin_deadband_{0.10f}, unitree_ang_deadband_{0.10f};
  float unitree_max_action_abs_{5.f};
  bool unitree_clip_by_torque_{true};
  double unitree_startup_sec_{0.2};
  float unitree_kp_factor_{1.f}, unitree_kd_factor_{1.f};
  float unitree_max_vx_{std::numeric_limits<float>::quiet_NaN()};
  float unitree_max_back_vx_{std::numeric_limits<float>::quiet_NaN()};
  float unitree_max_vy_{std::numeric_limits<float>::quiet_NaN()};
  float unitree_max_yaw_{std::numeric_limits<float>::quiet_NaN()};
  float unitree_default_max_vx_{2.f}, unitree_default_back_vx_{1.f}, unitree_default_max_vy_{1.f}, unitree_default_max_yaw_{1.f};
  double unitree_phase_period_{0.6}, unitree_phase_time_{0.0};

  std::vector<float> beyond_action_, beyond_ref_joint_pos_, beyond_ref_joint_vel_, beyond_ref_body_quat_w_;
  action_go::runtime::SuperMimicController supermimic_controller_;
  action_go::runtime::AllMimicController allmimic_controller_;
  action_go::runtime::MjlabCombatController mjlab_combat_controller_;
  bool supermimic_use_lowstate_tick_{false};
  bool supermimic_sim_lockstep_sequence_{false};
  bool supermimic_align_initial_yaw_{true};
  bool supermimic_stop_at_end_{false};
  bool supermimic_trace_{false};
  std::atomic<uint32_t> supermimic_command_sequence_{0};
  int beyond_ref_body_count_{0};
  int beyond_ref_anchor_idx_{0};
  int beyond_warmup_steps_{2};
  int beyond_initial_reference_step_{0};
  bool beyond_align_initial_yaw_{true};
  bool beyond_use_lowstate_tick_{false};
  bool beyond_sim_lockstep_sequence_{false};
  uint32_t beyond_policy_tick_ms_{20};
  int64_t beyond_motion_step_{0};
  std::array<int, 3> beyond_waist_joint_idx_{-1, -1, -1};
  std::array<float, 9> beyond_init_to_world_{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};

  int holosoma_motion_start_step_{0}, holosoma_motion_end_step_{0}, holosoma_current_timestep_{0};
  std::string holosoma_prepare_target_{"motion"};
  double holosoma_prepare_ramp_sec_{0.1}, holosoma_prepare_hold_sec_{0.0}, holosoma_policy_warmup_sec_{0.05};
  double holosoma_max_target_jump_{1.2};
  bool holosoma_stop_on_large_jump_{true}, holosoma_wall_time_timestep_{true};
  float holosoma_robot_yaw_offset_{0.f}, holosoma_motion_yaw_offset_{0.f};
  std::vector<float> holosoma_stiff_startup_pos_, holosoma_stiff_startup_kp_, holosoma_stiff_startup_kd_;
  std::vector<float> holosoma_last_policy_action_, holosoma_motion_joint_pos_, holosoma_motion_joint_vel_, holosoma_initial_motion_pos_;
  std::vector<float> holosoma_hold_q_, holosoma_hold_kp_, holosoma_hold_kd_, holosoma_prev_policy_target_q_;
  std::vector<std::string> holosoma_dof_names_;
  std::array<float, 4> holosoma_ref_quat_xyzw_{0.f, 0.f, 0.f, 1.f};
  bool holosoma_have_prev_policy_target_{false};
  int holosoma_large_upper_body_tracking_warnings_{0}, holosoma_large_target_step_warnings_{0};
  std::chrono::steady_clock::time_point holosoma_motion_clip_start_time_{};

  ProtoMotionCache proto_motion_;
  int proto_num_dofs_{0}, proto_num_bodies_{0}, proto_anchor_idx_{0}, proto_root_idx_{0};
  int proto_frame_{0};
  std::vector<int> proto_future_steps_;
  std::vector<std::string> proto_input_names_;
  std::unordered_map<std::string, std::string> proto_name_to_key_;
  std::vector<float> proto_prev_actions_;
  std::array<float, 4> proto_heading_offset_{0.f, 0.f, 0.f, 1.f};
  std::string proto_anchor_source_{"fk"};
  double proto_startup_interp_sec_{0.0};
  bool proto_heading_uninitialized_{true};
  bool proto_loop_motion_{false};

  int twist_policy_actions_{0};
  int twist_full_mimic_dim_{0};
  int twist_wrist_dim_{0};
  int twist_policy_mimic_dim_{0};
  int twist_proprio_dim_{0};
  int twist_obs_frame_dim_{0};
  int twist_input_dim_{0};
  int twist_obs_dim_{0}, twist_history_len_{0};
  std::string twist_input_source_{"offline-pkl"}, twist_realtime_transport_{"udp"};
  fs::path twist_motion_pkl_path_, twist_motion_cache_path_;
  std::string twist_root_, twist_prepare_script_, twist_steps_{"1"};
  std::string twist_udp_bind_{"127.0.0.1"}, twist_zmq_endpoint_{"tcp://127.0.0.1:15111"}, twist_zmq_topic_;
  std::string twist_redis_host_{"127.0.0.1"}, twist_redis_key_{"action_mimic_g1"};
  int twist_udp_port_{15110}, twist_redis_port_{6379};
  double twist_input_timeout_sec_{2.0}, twist_stale_sec_{0.25};
  float twist_action_scale_{0.5f}, twist_ang_vel_scale_{0.25f}, twist_dof_pos_scale_{1.0f}, twist_dof_vel_scale_{0.05f};
  bool twist_rebuild_motion_cache_{false}, twist_loop_motion_{false}, twist_require_realtime_{false};
  bool twist_zmq_bind_{true};
  StaleBehavior twist_stale_behavior_{StaleBehavior::kHoldThenStop};
  RealtimeInputStaleGate twist_stale_gate_;
  int64_t twist_stale_events_{0}, twist_stale_hold_ticks_{0}, twist_stale_stop_events_{0};
  std::vector<int> twist_body_motor_idx_, twist_wrist_motor_idx_, twist_ankle_policy_idx_;
  std::vector<float> twist_default_dof_pos_, twist_body_kp_, twist_body_kd_;
  std::vector<float> twist_wrist_default_, twist_wrist_kp_, twist_wrist_kd_;
  std::array<float, kTwistProtocolMimicDim> twist_default_mimic_obs_{};
  std::vector<float> twist_last_action_;
  std::deque<std::vector<float>> twist_history_;
  TwistOfflineMotionSource twist_offline_;
  TwistUdpReceiver twist_udp_;
  TwistZmqReceiver twist_zmq_;
  TwistRedisReceiver twist_redis_;
  std::optional<TwistMimicSample> twist_preview_sample_;

  int humanoid_gpt_obs_dim_{0};
  int humanoid_gpt_qpos_dim_{0};
  int humanoid_gpt_qvel_dim_{0};
  int humanoid_gpt_keypoint_count_{0};
  int humanoid_gpt_hmcp_header_size_{0};
  std::string humanoid_gpt_input_source_{"offline-npz"}, humanoid_gpt_realtime_transport_{"udp"};
  std::string humanoid_gpt_udp_bind_{"127.0.0.1"};
  int humanoid_gpt_udp_port_{51234};
  double humanoid_gpt_input_timeout_sec_{3.0}, humanoid_gpt_stale_sec_{0.25};
  bool humanoid_gpt_loop_reference_{false}, humanoid_gpt_require_realtime_{false};
  fs::path humanoid_gpt_convert_xml_path_;
  StaleBehavior humanoid_gpt_stale_behavior_{StaleBehavior::kHoldThenStop};
  RealtimeInputStaleGate humanoid_gpt_stale_gate_;
  int64_t humanoid_gpt_stale_events_{0}, humanoid_gpt_stale_hold_ticks_{0}, humanoid_gpt_stale_stop_events_{0};
  int humanoid_gpt_ref_frame_{0};
  bool humanoid_gpt_have_realtime_ref_{false};
  std::vector<std::string> humanoid_gpt_keypoint_names_;
  HumanoidGptReferenceCache humanoid_gpt_reference_cache_;
  HumanoidGptReferenceFrame humanoid_gpt_ref_curr_, humanoid_gpt_ref_next_;
  HumanoidGptUdpReceiver humanoid_gpt_udp_;
  HumanoidGptRefConverter humanoid_gpt_converter_;

  int sonic_encoder_obs_dim_{0}, sonic_decoder_obs_dim_{0}, sonic_token_dim_{0};
  int sonic_history_len_{0}, sonic_future_frames_{0}, sonic_future_step_{0}, sonic_zmq_header_size_{0};
  std::string sonic_checkpoint_{"default"};
  std::string sonic_input_source_{"offline-csv"};
  std::string sonic_zmq_endpoint_{"tcp://127.0.0.1:5556"}, sonic_zmq_topic_{"pose"};
  bool sonic_zmq_bind_{false}, sonic_require_realtime_{false}, sonic_loop_reference_{false};
  bool sonic_pico_control_{false}, sonic_use_planner_{false}, sonic_remote_stop_{false};
  double sonic_input_timeout_sec_{3.0}, sonic_stale_sec_{0.1}, sonic_stale_stop_sec_{0.5};
  double sonic_startup_ramp_sec_{3.0};
  std::string sonic_execution_provider_{"cpu"};
  int sonic_gpu_device_id_{0}, sonic_warmup_iters_{1};
  fs::path sonic_encoder_path_, sonic_decoder_path_, sonic_planner_path_, sonic_trt_cache_dir_;
  size_t sonic_trt_workspace_size_{1ULL << 30};
  bool sonic_trt_fp16_{false};
  uint64_t sonic_last_realtime_sequence_{0};
  uint64_t sonic_last_planner_sequence_{0}, sonic_last_control_sequence_{0};
  int64_t sonic_realtime_base_frame_{0};
  int64_t sonic_realtime_frame_step_{1};
  bool sonic_realtime_heading_reset_{false};
  int64_t sonic_reference_frame_{0}, sonic_stale_events_{0}, sonic_stale_hold_ticks_{0}, sonic_stale_stop_events_{0};
  bool sonic_hold_last_target_{false};
  SonicReferenceCache sonic_reference_cache_;
  SonicObservationBuilder sonic_observation_builder_;
  SonicPlannerConfig sonic_planner_config_;
  SonicPlanner sonic_planner_;
  SonicZmqReceiver sonic_zmq_;
  RealtimeInputStaleGate sonic_stale_gate_;
  OnnxRunner sonic_encoder_, sonic_decoder_, sonic_planner_runner_;

  CommandSafetyConfig command_safety_config_;
  CommandSafetyFilter safety_filter_;
  CommandSafetyLimits safety_limits_;
  RuntimeEventLogger event_log_;
  CommandTimingReporter timing_reporter_;
  ActivationBlend activation_blend_;
  OnnxRunner onnx_;
  unitree::robot::ChannelPublisherPtr<LowCmd> publisher_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  mutable std::mutex state_mtx_;
  RobotState state_;
  uint32_t dry_run_command_crc_{0};
  std::atomic<int64_t> active_publish_count_{0};
  std::atomic<int64_t> last_active_publish_ns_{0};
  std::atomic<double> max_active_publish_interval_ms_{0.0};
  std::atomic<bool> connected_{false};
  bool first_command_logged_{false};
  bool first_hold_logged_{false};
  bool first_active_command_logged_{false};
  bool prestart_hold_suppressed_logged_{false};
  bool start_file_seen_logged_{false};
  bool first_preview_candidate_logged_{false};
  bool preview_active_consistency_logged_{false};
  bool activation_blend_bypass_logged_{false};
  std::vector<float> preview_candidate_q_, preview_candidate_kp_, preview_candidate_kd_;
};

}  // namespace action_go::runtime::generic_policy_runtime
