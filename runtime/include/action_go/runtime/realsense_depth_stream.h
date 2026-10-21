#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace action_go::runtime {

struct RealSenseDepthConfig {
  int width{0};
  int height{0};
  int fps{0};
  int output_width{0};
  int output_height{0};
  int min_frames{0};
  int crop_top{0};
  int crop_bottom{0};
  int crop_left{0};
  int crop_right{0};
  float horizontal_fov_deg{0.f};
  float vertical_fov_deg{0.f};
  float invalid_fill_m{0.f};
  float min_valid_fraction{0.f};
  double startup_timeout_sec{0.0};
  double timeout_sec{0.0};
};

class RealSenseDepthStream {
 public:
  explicit RealSenseDepthStream(const RealSenseDepthConfig& config);
  ~RealSenseDepthStream();
  std::vector<float> observation();
  uint64_t sequence() const;
  double age_sec() const;
  std::string stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}
