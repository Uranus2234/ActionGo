#pragma once

#include "action_go/runtime/bfm_context.h"
#include "action_go/runtime/bfm_observation.h"
#include "action_go/runtime/onnx_runner.h"
#include "action_go/runtime/realsense_depth_stream.h"

#include <yaml-cpp/yaml.h>

#include <array>
#include <deque>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace action_go::runtime {

struct PbfmSpec {
  BfmContext context;
  BfmContextSequence context_sequence;
  std::filesystem::path context_path;
  std::string context_source{"cache"};
  std::string context_udp_bind;
  int context_udp_port{0};
  double context_timeout_sec{0.0};
  double context_startup_timeout_sec{0.0};
  std::string depth_source;
  std::string shm_name;
  std::filesystem::path depth_file;
  double depth_timeout_sec{0.0};
  RealSenseDepthConfig realsense;
  std::string command_source{"context"};
  float max_forward_speed{0.f};
  float max_backward_speed{0.f};
  float max_lateral_speed{0.f};
  float max_yaw_rate{0.f};
  float max_heading_lead{0.f};
  float max_translation_speed{0.f};
  bool proportional_speed{false};
  BfmContext idle_context;
  std::filesystem::path idle_context_path;
  float command_deadband{0.f};
  std::unordered_map<std::string, BfmContext> directional_contexts;
  std::unordered_map<std::string, BfmContextSequence> directional_sequences;
  std::unordered_map<std::string, BfmContextSequence> turn_sequences;

  static PbfmSpec load(const YAML::Node& root, const std::filesystem::path& spec_path);
  BfmContext movement_context(float forward_speed, float lateral_speed, size_t reference_step = 0) const;
  BfmContext turning_context(float yaw_rate, size_t reference_step) const;
  BfmContext blend_with_idle(BfmContext context, float strength) const;
};

class PbfmDepthHistory {
 public:
  std::vector<float> append(const std::vector<float>& optical_z_m);

 private:
  std::deque<std::vector<float>> frames_;
};

class PbfmHeadingController {
 public:
  void reset(float yaw);
  float update(float axis, float max_rate, float max_lead, float current_yaw, float dt);

 private:
  float target_yaw_{0.f};
  bool was_turning_{false};
};

struct PbfmResult {
  std::vector<float> action;
  std::vector<float> estimated_velocity;
};

class PbfmPolicy {
 public:
  static void validate_model(const OnnxRunner& runner);
  static std::unordered_map<std::string, OnnxRunner::TensorInput> inputs(
      const BfmObservation& observation, const BfmContext& context,
      const std::array<float, 2>& heading, const std::vector<float>& depth);
  static PbfmResult infer(OnnxRunner& runner, const BfmObservation& observation,
                          const BfmContext& context, const std::array<float, 2>& heading,
                          const std::vector<float>& depth);
};

}
