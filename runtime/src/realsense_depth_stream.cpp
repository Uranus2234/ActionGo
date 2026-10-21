#include "action_go/runtime/realsense_depth_stream.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef ACTION_GO_WITH_REALSENSE
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/imgproc.hpp>
#endif

namespace action_go::runtime {

class RealSenseDepthStream::Impl {
 public:
  explicit Impl(const RealSenseDepthConfig& config) : config_(config) {
#ifndef ACTION_GO_WITH_REALSENSE
    throw std::runtime_error("RealSense depth requires librealsense2; rebuild with ACTION_GO_REQUIRE_REALSENSE=ON");
#else
    try {
      initialize();
    } catch (...) {
      stop();
      throw;
    }
#endif
  }

  ~Impl() { stop(); }

  std::vector<float> observation() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_.empty()) throw std::runtime_error(error_);
    if (sequence_ == 0 || elapsed() > config_.timeout_sec) {
      throw std::runtime_error("RealSense depth is missing or stale; " + stats_locked());
    }
    return latest_;
  }

  uint64_t sequence() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sequence_;
  }

  double age_sec() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return elapsed();
  }

  std::string stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_locked();
  }

 private:
  double elapsed() const {
    if (sequence_ == 0) return -1.0;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - stamp_).count();
  }

  std::string stats_locked() const {
    const double received_age = received_frames_ == 0 ? -1.0 :
        std::chrono::duration<double>(std::chrono::steady_clock::now() - received_stamp_).count();
    return "source=realsense seq=" + std::to_string(sequence_) +
           " age_sec=" + std::to_string(elapsed()) +
           " received_age_sec=" + std::to_string(received_age) +
           " received_frames=" + std::to_string(received_frames_) +
           " rejected_frames=" + std::to_string(rejected_frames_) +
           " last_frame_number=" + std::to_string(last_frame_number_) +
           " policy crop valid_fraction=" + std::to_string(latest_valid_fraction_) +
           " required=" + std::to_string(config_.min_valid_fraction);
  }

  void stop() {
    stopped_.store(true);
    condition_.notify_all();
#ifdef ACTION_GO_WITH_REALSENSE
    if (worker_.joinable()) worker_.join();
    if (pipeline_started_) {
      pipeline_started_ = false;
      try {
        pipeline_.stop();
      } catch (const std::exception& error) {
        std::cerr << "[depth] RealSense stop failed: " << error.what() << '\n';
      }
    }
#endif
  }

#ifdef ACTION_GO_WITH_REALSENSE
  void initialize() {
    const auto& config = config_;
    rs2::config request;
    request.enable_stream(RS2_STREAM_DEPTH, config.width, config.height, RS2_FORMAT_Z16, config.fps);
    const auto profile = pipeline_.start(request);
    pipeline_started_ = true;
    scale_ = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
    const auto intrinsics = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    map_x_.create(config.output_height, config.output_width, CV_32FC1);
    map_y_.create(config.output_height, config.output_width, CV_32FC1);
    const float radians = static_cast<float>(std::acos(-1.0) / 180.0);
    const float focal_x = config.output_width / (2.f * std::tan(config.horizontal_fov_deg * radians / 2.f));
    const float focal_y = config.output_height / (2.f * std::tan(config.vertical_fov_deg * radians / 2.f));
    for (int row = 0; row < config.output_height; ++row) {
      for (int column = 0; column < config.output_width; ++column) {
        const float ray[3]{(column + 0.5f - config.output_width / 2.f) / focal_x,
                           (row + 0.5f - config.output_height / 2.f) / focal_y, 1.f};
        float pixel[2];
        rs2_project_point_to_pixel(pixel, &intrinsics, ray);
        const bool required = row >= config.crop_top && row < config.output_height - config.crop_bottom &&
                              column >= config.crop_left && column < config.output_width - config.crop_right;
        const bool covered = std::isfinite(pixel[0]) && std::isfinite(pixel[1]) && pixel[0] >= 0.f &&
                             pixel[1] >= 0.f && pixel[0] <= intrinsics.width - 1.f &&
                             pixel[1] <= intrinsics.height - 1.f;
        if (required && !covered) {
          throw std::runtime_error("RealSense calibrated field of view does not cover the requested depth rays");
        }
        map_x_.at<float>(row, column) = covered ? pixel[0] : -1.f;
        map_y_.at<float>(row, column) = covered ? pixel[1] : -1.f;
      }
    }
    worker_ = std::thread(&Impl::capture, this);
    std::unique_lock<std::mutex> lock(mutex_);
    const bool available = condition_.wait_for(lock, std::chrono::duration<double>(config.startup_timeout_sec),
        [&] { return sequence_ >= static_cast<uint64_t>(config.min_frames) || !error_.empty(); });
    if (!available || !error_.empty()) {
      const std::string reason = error_.empty() ?
          "RealSense did not provide enough valid depth frames before ready; " + stats_locked() : error_;
      throw std::runtime_error(reason);
    }
    std::cout << "[depth] RealSense " << config.width << 'x' << config.height << '@' << config.fps
              << " -> calibrated optical-Z " << config.output_width << 'x' << config.output_height
              << " fx=" << intrinsics.fx << " fy=" << intrinsics.fy
              << " ppx=" << intrinsics.ppx << " ppy=" << intrinsics.ppy << '\n';
  }

  void capture() {
    try {
      while (!stopped_.load()) {
        rs2::frameset frames;
        if (!pipeline_.poll_for_frames(&frames)) {
          std::unique_lock<std::mutex> lock(mutex_);
          condition_.wait_for(lock, std::chrono::milliseconds(5), [&] { return stopped_.load(); });
          continue;
        }
        const auto depth = frames.get_depth_frame();
        if (!depth) continue;
        const auto received_at = std::chrono::steady_clock::now();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          received_stamp_ = received_at;
          last_frame_number_ = depth.get_frame_number();
          ++received_frames_;
        }
        const cv::Mat raw(config_.height, config_.width, CV_16UC1, const_cast<void*>(depth.get_data()),
                          depth.get_stride_in_bytes());
        cv::Mat valid_rays;
        cv::remap(raw != 0, valid_rays, map_x_, map_y_, cv::INTER_NEAREST,
                  cv::BORDER_CONSTANT, cv::Scalar(0));
        const cv::Rect crop(config_.crop_left, config_.crop_top,
                            config_.output_width - config_.crop_left - config_.crop_right,
                            config_.output_height - config_.crop_top - config_.crop_bottom);
        const float valid_fraction = static_cast<float>(cv::countNonZero(valid_rays(crop))) / crop.area();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          latest_valid_fraction_ = valid_fraction;
          if (valid_fraction < config_.min_valid_fraction) ++rejected_frames_;
        }
        if (valid_fraction < config_.min_valid_fraction) continue;
        cv::Mat metres;
        raw.convertTo(metres, CV_32FC1, scale_);
        metres.setTo(config_.invalid_fill_m, raw == 0);
        cv::Mat projected;
        cv::remap(metres, projected, map_x_, map_y_, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT, cv::Scalar(config_.invalid_fill_m));
        std::vector<float> output(projected.ptr<float>(), projected.ptr<float>() + projected.total());
        {
          std::lock_guard<std::mutex> lock(mutex_);
          latest_ = std::move(output);
          stamp_ = received_at;
          ++sequence_;
        }
        condition_.notify_all();
      }
    } catch (const std::exception& error) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = std::string("RealSense capture failed: ") + error.what();
      }
      condition_.notify_all();
    }
  }

  rs2::pipeline pipeline_;
  bool pipeline_started_{false};
  cv::Mat map_x_;
  cv::Mat map_y_;
  float scale_{0.f};
#endif
  RealSenseDepthConfig config_;
  float latest_valid_fraction_{0.f};
  std::atomic<bool> stopped_{false};
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<float> latest_;
  std::string error_;
  uint64_t sequence_{0};
  uint64_t received_frames_{0};
  uint64_t rejected_frames_{0};
  uint64_t last_frame_number_{0};
  std::chrono::steady_clock::time_point received_stamp_{};
  std::chrono::steady_clock::time_point stamp_{};
};

RealSenseDepthStream::RealSenseDepthStream(const RealSenseDepthConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
RealSenseDepthStream::~RealSenseDepthStream() = default;
std::vector<float> RealSenseDepthStream::observation() { return impl_->observation(); }
uint64_t RealSenseDepthStream::sequence() const { return impl_->sequence(); }
double RealSenseDepthStream::age_sec() const { return impl_->age_sec(); }
std::string RealSenseDepthStream::stats() const {
  return impl_->stats();
}

}
