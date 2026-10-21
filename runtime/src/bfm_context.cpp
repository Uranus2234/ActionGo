#include "action_go/runtime/bfm_context.h"
#include "action_go/runtime/yaml_contract.h"

#include <cmath>
#include <stdexcept>

namespace action_go::runtime {

BfmContext BfmContext::load(const std::filesystem::path& path, size_t latent_dim) {
  namespace yamlc = yaml_contract;
  const auto root = YAML::LoadFile(path.string());
  if (yamlc::required_scalar<std::string>(root, "contract") != "reward_goal_yaw_command_v1") {
    throw std::runtime_error("unsupported BFM fixed context contract");
  }
  return from_yaml(root, latent_dim);
}

BfmContext BfmContext::from_yaml(const YAML::Node& root, size_t latent_dim) {
  namespace yamlc = yaml_contract;
  BfmContext context;
  context.latent = yamlc::required_vector_size<float>(root, "z", latent_dim);
  const auto heading = yamlc::required_vector_size<float>(root, "heading_relative_xy", 2);
  context.heading_relative = {heading[0], heading[1]};
  context.heading_valid = yamlc::required_scalar<bool>(root, "heading_valid");
  const auto velocity = yamlc::required_vector_size<float>(root, "velocity_command", 3);
  context.velocity_command = {velocity[0], velocity[1], velocity[2]};
  float norm_squared = 0.f;
  for (float value : context.latent) {
    if (!std::isfinite(value)) throw std::runtime_error("BFM context latent contains NaN/Inf");
    norm_squared += value * value;
  }
  if (norm_squared <= 0.f) throw std::runtime_error("BFM context latent must be nonzero");
  for (float value : heading) {
    if (!std::isfinite(value)) throw std::runtime_error("BFM context heading contains NaN/Inf");
  }
  const float heading_norm = std::hypot(heading[0], heading[1]);
  if (context.heading_valid && std::abs(heading_norm - 1.f) > 1e-4f) {
    throw std::runtime_error("BFM context heading must be a unit XY vector");
  }
  for (float value : velocity) {
    if (!std::isfinite(value)) throw std::runtime_error("BFM velocity command contains NaN/Inf");
  }
  if (velocity[2] != 0.f && velocity[2] != 1.f) {
    throw std::runtime_error("BFM velocity command validity must be 0 or 1");
  }
  if (velocity[2] == 0.f && (velocity[0] != 0.f || velocity[1] != 0.f)) {
    throw std::runtime_error("invalid BFM velocity command must be [0,0,0]");
  }
  return context;
}

std::array<float, 2> BfmContext::heading_input(float initial_yaw, float current_yaw) const {
  if (!heading_valid) return {0.f, 0.f};
  const float error = initial_yaw + std::atan2(heading_relative[1], heading_relative[0]) - current_yaw;
  return {1.f - std::cos(error), std::sin(error)};
}

BfmContext BfmContext::aligned(float reference_start_yaw, float initial_yaw, float current_yaw) const {
  auto context = *this;
  const float heading = std::atan2(heading_relative[1], heading_relative[0]) - reference_start_yaw;
  context.heading_relative = {std::cos(heading), std::sin(heading)};
  const float rotation = initial_yaw - reference_start_yaw - current_yaw;
  context.velocity_command[0] = std::cos(rotation) * velocity_command[0] - std::sin(rotation) * velocity_command[1];
  context.velocity_command[1] = std::sin(rotation) * velocity_command[0] + std::cos(rotation) * velocity_command[1];
  return context;
}

BfmContextSequence BfmContextSequence::load(const std::filesystem::path& path, size_t latent_dim) {
  namespace yamlc = yaml_contract;
  const auto root = YAML::LoadFile(path.string());
  if (yamlc::required_scalar<std::string>(root, "contract") != "tracking_heading0_yaw_velocity_v1") {
    throw std::runtime_error("unsupported BFM context sequence contract");
  }
  BfmContextSequence sequence;
  sequence.dt = yamlc::required_scalar<double>(root, "dt");
  if (!std::isfinite(sequence.dt) || sequence.dt <= 0.0) {
    throw std::runtime_error("BFM context sequence dt must be positive");
  }
  const auto frames = root["frames"];
  if (!frames.IsSequence() || frames.size() == 0) {
    throw std::runtime_error("BFM context sequence requires nonempty frames");
  }
  for (const auto& frame : frames) {
    auto converted = YAML::Clone(frame);
    const auto velocity = yamlc::required_vector_size<float>(frame, "reference_velocity_heading0_xy", 2);
    const bool valid = yamlc::required_scalar<bool>(frame, "velocity_valid");
    converted["velocity_command"] = std::vector<float>{velocity[0], velocity[1], valid ? 1.f : 0.f};
    sequence.frames.push_back(BfmContext::from_yaml(converted, latent_dim));
  }
  return sequence;
}

BfmContext BfmContextSequence::at(size_t step, float initial_yaw, float current_yaw) const {
  auto context = frames.at(step);
  const float cosine = std::cos(initial_yaw - current_yaw);
  const float sine = std::sin(initial_yaw - current_yaw);
  const float vx = context.velocity_command[0];
  const float vy = context.velocity_command[1];
  context.velocity_command[0] = cosine * vx - sine * vy;
  context.velocity_command[1] = sine * vx + cosine * vy;
  return context;
}

}
