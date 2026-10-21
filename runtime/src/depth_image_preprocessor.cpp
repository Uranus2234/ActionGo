#include "action_go/runtime/depth_image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace action_go::runtime {
namespace {

float cubic(float distance) {
  distance = std::abs(distance);
  if (distance < 1.f) return ((1.5f * distance - 2.5f) * distance) * distance + 1.f;
  if (distance < 2.f) return ((-0.5f * distance + 2.5f) * distance - 4.f) * distance + 2.f;
  return 0.f;
}

}

std::vector<float> preprocess_php_depth(const std::vector<float>& depth_m) {
  if (depth_m.size() != 60 * 106) throw std::runtime_error("PHP raw depth must be 60x106 metres");
  std::vector<float> output(58 * 87);
  const float scale = 98.f / 87.f;
  for (int column = 0; column < 87; ++column) {
    const float center = (column + 0.5f) * scale;
    const int begin = std::max(0, static_cast<int>(center - 2.f * scale + 0.5f));
    const int end = std::min(98, static_cast<int>(center + 2.f * scale + 0.5f));
    std::vector<float> weights;
    float total = 0.f;
    for (int source = begin; source < end; ++source) {
      weights.push_back(cubic((source + 0.5f - center) / scale));
      total += weights.back();
    }
    for (float& weight : weights) weight /= total;
    for (int row = 0; row < 58; ++row) {
      float value = 0.f;
      for (int source = begin; source < end; ++source) {
        const float raw = depth_m[(row + 2) * 106 + source + 4];
        if (!std::isfinite(raw)) throw std::runtime_error("PHP raw depth contains NaN/Inf");
        value += std::clamp(raw, 0.3f, 3.f) * weights[source - begin];
      }
      value = std::min(value, 3.f);
      if (value < 0.15f) value = 3.f;
      output[row * 87 + column] = (value - 0.3f) / 2.7f - 0.5f;
    }
  }
  return output;
}

std::vector<float> preprocess_pbfm_depth(const std::vector<float>& optical_z_m) {
  if (optical_z_m.size() != 36 * 64) throw std::runtime_error("P-BFM raw optical-Z must be 36x64 metres");
  std::vector<float> cropped(18 * 32);
  for (int row = 0; row < 18; ++row) {
    for (int column = 0; column < 32; ++column) {
      const float raw = optical_z_m[(row + 18) * 64 + column + 16];
      cropped[row * 32 + column] = std::isfinite(raw) && raw > 0.f ? std::min(raw, 2.5f) : 2.5f;
    }
  }
  const float edge = std::exp(-0.5f);
  const float total = 1.f + 2.f * edge;
  const float weights[3]{edge / total, 1.f / total, edge / total};
  const auto reflect = [](int index, int size) {
    return index < 0 ? -index : index >= size ? 2 * size - 2 - index : index;
  };
  std::vector<float> output(18 * 32, 0.f);
  for (int row = 0; row < 18; ++row) {
    for (int column = 0; column < 32; ++column) {
      float blurred = 0.f;
      for (int offset_row = 0; offset_row < 3; ++offset_row) {
        for (int offset_column = 0; offset_column < 3; ++offset_column) {
          blurred += (weights[offset_row] * weights[offset_column]) *
              cropped[reflect(row + offset_row - 1, 18) * 32 + reflect(column + offset_column - 1, 32)];
        }
      }
      output[row * 32 + column] = std::nearbyint(std::clamp(blurred, 0.f, 2.5f) / 2.5f * 255.f) / 255.f;
    }
  }
  return output;
}

}
