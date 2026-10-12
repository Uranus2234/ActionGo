#pragma once

#include "action_go/runtime/policy_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace action_go::runtime {

inline constexpr int kSonicMotorCount = 29;
inline constexpr int kSonicSmplFrames = kSonicSmplReferenceFrames;

enum class SonicObservationLayout {
  kDefault,
  kV11,
};

struct SonicReferenceFrame {
  std::array<float, kSonicMotorCount> joint_pos{};
  std::array<float, kSonicMotorCount> joint_vel{};
  std::array<float, kSonicSmplJointCount * 3> smpl_joints{};
  std::array<float, 4> body_quat_wxyz{1.f, 0.f, 0.f, 0.f};
  float root_z{0.f};
  int64_t frame_index{0};
  int encoder_mode{0};
  bool has_explicit_frame_index{false};
};

struct SonicRobotFrame {
  std::array<float, 4> base_quat_wxyz{1.f, 0.f, 0.f, 0.f};
  std::array<float, 3> base_ang_vel{};
  std::array<float, kSonicMotorCount> joint_pos{};
  std::array<float, kSonicMotorCount> joint_vel{};
  std::array<float, kSonicMotorCount> last_action{};
};

class SonicReferenceCache {
 public:
  void load_csv_directory(const std::filesystem::path& directory);
  void replace(std::vector<SonicReferenceFrame> frames);
  void append(std::vector<SonicReferenceFrame> frames);
  void merge_tail(std::vector<SonicReferenceFrame> frames);
  void discard_before(int64_t frame_index);
  const SonicReferenceFrame& frame_clamped(int64_t index) const;
  int64_t local_index_for_frame(int64_t frame_index) const;
  int64_t first_frame_index() const;
  int64_t last_frame_index() const;
  int64_t frame_step() const;
  size_t size() const { return frames_.size(); }
  bool empty() const { return frames_.empty(); }

 private:
  std::vector<SonicReferenceFrame> frames_;
};

class SonicObservationBuilder {
 public:
  void set_layout(SonicObservationLayout layout) { layout_ = layout; }
  void reset(const SonicRobotFrame& initial_robot, const SonicReferenceFrame& initial_reference);
  void reset_heading(const SonicRobotFrame& robot, const SonicReferenceFrame& reference);
  void push_robot_frame(const SonicRobotFrame& frame);

  std::vector<float> build_encoder_observation(const SonicReferenceCache& reference,
                                                int64_t current_frame,
                                                int future_frames,
                                                int future_step) const;
  std::vector<float> build_decoder_observation(const std::vector<float>& token) const;

 private:
  SonicObservationLayout layout_{SonicObservationLayout::kDefault};
  std::vector<SonicRobotFrame> history_;
  std::array<float, 4> initial_robot_heading_{1.f, 0.f, 0.f, 0.f};
  std::array<float, 4> initial_reference_heading_{1.f, 0.f, 0.f, 0.f};
};

const std::array<int, kSonicMotorCount>& sonic_policy_to_hardware();
const std::array<int, kSonicMotorCount>& sonic_hardware_to_policy();

std::array<float, kSonicMotorCount> sonic_robot_to_policy(
    const std::vector<float>& hardware_values,
    const std::vector<float>& default_q,
    bool subtract_default);

std::vector<float> sonic_decode_target(const std::vector<float>& policy_action,
                                       const std::vector<float>& default_q,
                                       const std::vector<float>& action_scale);

std::vector<float> sonic_startup_target(const std::vector<float>& measured_q,
                                        const std::vector<float>& default_q,
                                        float alpha);

std::vector<SonicReferenceFrame> parse_sonic_zmq_v1_packet(
    const std::vector<uint8_t>& packet, const std::string& topic,
    std::string* layout_signature = nullptr);

// Parses the upstream GEAR-SONIC packed pose protocols: v1 G1 reference and
// v2/v3 SMPL reference. The v1-named entrypoint above remains as a compatible
// alias for existing callers.
std::vector<SonicReferenceFrame> parse_sonic_zmq_packet(
    const std::vector<uint8_t>& packet, const std::string& topic,
    std::string* layout_signature = nullptr);

}  // namespace action_go::runtime
