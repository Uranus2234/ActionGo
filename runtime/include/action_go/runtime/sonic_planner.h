#pragma once

#include "action_go/runtime/onnx_runner.h"
#include "action_go/runtime/sonic_policy.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace action_go::runtime {

inline constexpr int kSonicPlannerMaxFrames = 64;
inline constexpr int kSonicPlannerModeCountV2 = 27;

enum class SonicLocomotionMode : int64_t {
  kIdle = 0,
  kSlowWalk = 1,
  kWalk = 2,
  kRun = 3,
  kSquat = 4,
  kKneelTwoLegs = 5,
  kKneel = 6,
  kLyingFaceDown = 7,
  kCrawling = 8,
  kIdleBoxing = 9,
  kWalkBoxing = 10,
  kLeftPunch = 11,
  kRightPunch = 12,
  kRandomPunch = 13,
  kElbowCrawling = 14,
  kLeftHook = 15,
  kRightHook = 16,
  kForwardJump = 17,
  kStealthWalk = 18,
  kInjuredWalk = 19,
  kLedgeWalking = 20,
  kObjectCarrying = 21,
  kStealthWalk2 = 22,
  kHappyDanceWalk = 23,
  kZombieWalk = 24,
  kGunWalk = 25,
  kScareWalk = 26,
};

struct SonicPlannerCommand {
  int64_t mode{0};
  std::array<float, 3> movement{};
  std::array<float, 3> facing{1.f, 0.f, 0.f};
  float speed{-1.f};
  float height{-1.f};
};

struct SonicControlCommand {
  bool start{false};
  bool stop{false};
  bool planner{true};
  bool has_delta_heading{false};
  float delta_heading{0.f};
};

struct SonicPlannerConfig {
  int version{2};
  int lookahead_steps{2};
  int blend_frames{8};
  int min_output_frames{24};
  int max_output_frames{kSonicPlannerMaxFrames};
  float default_height{0.788740f};
  int64_t random_seed{1234};
  std::array<int64_t, 11> allowed_pred_num_tokens{0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0};
};

struct SonicPlannerMotionFrame {
  std::array<float, 3> root_position{};
  std::array<float, 4> body_quat_wxyz{1.f, 0.f, 0.f, 0.f};
  std::array<float, kSonicMotorCount> joint_pos{};
  std::array<float, kSonicMotorCount> joint_vel{};
};

SonicPlannerCommand parse_sonic_planner_packet(
    const std::vector<uint8_t>& packet, const std::string& topic = "planner",
    std::string* layout_signature = nullptr);

SonicControlCommand parse_sonic_command_packet(
    const std::vector<uint8_t>& packet, const std::string& topic = "command",
    std::string* layout_signature = nullptr);

bool sonic_planner_mode_is_static(int64_t mode);
double sonic_planner_replan_period_sec(int64_t mode);
const char* sonic_planner_mode_name(int64_t mode);

class SonicPlanner {
 public:
  explicit SonicPlanner(SonicPlannerConfig config = {});

  void initialize(OnnxRunner& runner,
                  const std::vector<float>& hardware_joint_positions);
  bool should_replan(const SonicPlannerCommand& command,
                     std::chrono::steady_clock::time_point now) const;
  void replan(OnnxRunner& runner, const SonicPlannerCommand& command,
              std::chrono::steady_clock::time_point now);
  void advance();
  void rewind() { current_frame_ = 0; }

  bool initialized() const { return initialized_; }
  int current_frame() const { return current_frame_; }
  int generated_frame() const { return generated_frame_; }
  size_t size() const { return motion_.size(); }
  const SonicPlannerCommand& last_command() const { return last_command_; }
  const std::vector<SonicPlannerMotionFrame>& motion() const { return motion_; }
  std::vector<SonicReferenceFrame> reference_frames() const;

  std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim>
  initial_context(const std::vector<float>& hardware_joint_positions) const;
  std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim>
  context_from_motion() const;
  std::vector<SonicPlannerMotionFrame> resample_output(
      const std::vector<float>& qpos, int32_t num_pred_frames) const;
  void merge_generated(std::vector<SonicPlannerMotionFrame> generated,
                       int generated_frame);

 private:
  std::vector<SonicPlannerMotionFrame> run(
      OnnxRunner& runner,
      const std::array<float, kSonicPlannerContextFrames * kSonicPlannerQposDim>& context,
      const SonicPlannerCommand& command) const;

  SonicPlannerConfig config_;
  std::vector<SonicPlannerMotionFrame> motion_;
  SonicPlannerCommand last_command_{};
  std::chrono::steady_clock::time_point last_plan_time_{};
  int current_frame_{0};
  int generated_frame_{0};
  bool initialized_{false};
};

}  // namespace action_go::runtime
