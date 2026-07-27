#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace action_go::runtime {

enum class PreviewLifecycleDecision {
  kCommit,
  kDiscard,
  kTimeout,
  kStopped,
};

inline void write_lifecycle_file(const std::string& path,
                                 const std::string& value) {
  if (path.empty()) return;
  const std::filesystem::path output(path);
  if (output.has_parent_path()) {
    std::filesystem::create_directories(output.parent_path());
  }
  const std::filesystem::path temporary = output.string() + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::out | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to write lifecycle file: " + path);
    stream << value << '\n';
  }
  std::filesystem::rename(temporary, output);
}

inline void validate_preview_lifecycle(const std::string& preview_file,
                                       const std::string& commit_file,
                                       const std::string& discard_file,
                                       double timeout_sec,
                                       const std::string& candidate_topic) {
  if (preview_file.empty() || commit_file.empty()) {
    throw std::runtime_error(
        "transactional preview requires --preview-file and --commit-file");
  }
  if (!std::isfinite(timeout_sec) || timeout_sec <= 0.0) {
    throw std::runtime_error("--preview-timeout-sec must be positive");
  }
  if (candidate_topic == "rt/user_lowcmd" || candidate_topic == "rt/lowcmd") {
    throw std::runtime_error(
        "transactional preview requires a private candidate topic");
  }
  if (std::filesystem::exists(commit_file) ||
      (!discard_file.empty() && std::filesystem::exists(discard_file))) {
    throw std::runtime_error(
        "stale preview lifecycle input exists; remove commit/discard files before launch");
  }
  std::error_code error;
  std::filesystem::remove(preview_file, error);
}

template <typename StopPredicate, typename PublishCandidate>
PreviewLifecycleDecision wait_preview_lifecycle(
    const std::string& preview_file,
    const std::string& commit_file,
    const std::string& discard_file,
    double timeout_sec,
    int publish_hz,
    StopPredicate stop_requested,
    PublishCandidate publish_candidate) {
  write_lifecycle_file(preview_file, "preview_ready");
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_sec));
  const auto tick = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / std::max(1, publish_hz)));
  auto next = std::chrono::steady_clock::now();
  while (!stop_requested()) {
    if (std::filesystem::exists(commit_file)) {
      return PreviewLifecycleDecision::kCommit;
    }
    if (!discard_file.empty() && std::filesystem::exists(discard_file)) {
      return PreviewLifecycleDecision::kDiscard;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return PreviewLifecycleDecision::kTimeout;
    }
    publish_candidate();
    next += tick;
    if (next < std::chrono::steady_clock::now()) next = std::chrono::steady_clock::now();
    std::this_thread::sleep_until(next);
  }
  return PreviewLifecycleDecision::kStopped;
}

}  // namespace action_go::runtime
