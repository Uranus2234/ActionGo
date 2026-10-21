#pragma once

#include <yaml-cpp/yaml.h>

#include <array>
#include <filesystem>
#include <vector>

namespace action_go::runtime {

struct BfmContext {
  std::vector<float> latent;
  std::array<float, 2> heading_relative{1.f, 0.f};
  bool heading_valid{false};
  std::array<float, 3> velocity_command{0.f, 0.f, 0.f};

  static BfmContext load(const std::filesystem::path& path, size_t latent_dim);
  static BfmContext from_yaml(const YAML::Node& root, size_t latent_dim);
  std::array<float, 2> heading_input(float initial_yaw, float current_yaw) const;
  BfmContext aligned(float reference_start_yaw, float initial_yaw, float current_yaw) const;
};

struct BfmContextSequence {
  std::vector<BfmContext> frames;
  double dt{0.0};

  static BfmContextSequence load(const std::filesystem::path& path, size_t latent_dim);
  BfmContext at(size_t step, float initial_yaw, float current_yaw) const;
};

}
