#pragma once

#include <nlohmann/json.hpp>
#include "action_go/runtime/log_provenance.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

namespace action_go::runtime {

class RuntimeSession {
 public:
  RuntimeSession(const std::string& event_path, const std::string& config,
                 const std::string& remote_map, const std::string& net, bool dry_run) {
    namespace fs = std::filesystem;
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream prefix;
    prefix << std::put_time(&utc, "%Y%m%d_%H%M%S") << '_';
    const char* configured = std::getenv("ACTION_GO_LOG_DIR");
    fs::path root = configured && *configured ? configured : "logs";
    fs::create_directories(root);
    std::string pattern = (fs::absolute(root) / (prefix.str() + "XXXXXX")).string();
    if (!mkdtemp(pattern.data())) throw std::runtime_error("cannot create runtime log directory");
    directory_ = pattern;
    fs::create_directory(directory_ / "actions");
    event_path_ = event_path.empty() ? (directory_ / "events.jsonl").string() : event_path;
    const auto run_id = directory_.filename().string();
    setenv("ACTION_GO_RUN_ID", run_id.c_str(), 1);
    setenv("ACTION_GO_SESSION_DIR", directory_.c_str(), 1);
    std::error_code link_error;
    fs::remove(root / "latest", link_error);
    fs::create_directory_symlink(directory_, root / "latest", link_error);
    const char* code_revision = std::getenv("ACTION_GO_GIT_SHA");
    nlohmann::json metadata{{"application", "ActionGo"}, {"pid", getpid()}, {"run_id", run_id}, {"net", net}, {"dry_run", dry_run},
                            {"config", config}, {"remote_map", remote_map},
                            {"events", fs::absolute(event_path_).string()},
                            {"config_sha256", log_file_sha256(config)},
                            {"remote_map_sha256", log_file_sha256(remote_map)}};
    metadata["git_sha"] = code_revision ? code_revision : "unknown";
    for (const auto& entry : {std::make_pair(config, "actions.yaml"),
                              std::make_pair(remote_map, "remote.yaml")}) {
      if (!entry.first.empty() && fs::is_regular_file(entry.first)) {
        fs::copy_file(entry.first, directory_ / entry.second);
      }
    }
    std::ofstream(directory_ / "session.json") << metadata.dump(2) << '\n';
    notices_.set_context(action_lifecycle_context_from_environment());
    notices_.open((directory_ / "session_events.jsonl").string());
    const char* mode = std::getenv("ACTION_GO_CONSOLE");
    mode_ = mode ? mode : "normal";
    if (mode_ != "normal" && mode_ != "quiet" && mode_ != "debug") {
      throw std::runtime_error("ACTION_GO_CONSOLE must be normal, quiet or debug");
    }
    try {
      console_ = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
      saved_error_ = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
      if (console_ < 0 || saved_error_ < 0) throw std::runtime_error("cannot preserve terminal descriptors");
      tty_ = isatty(console_);
      int descriptors[2];
      if (pipe2(descriptors, O_CLOEXEC) != 0) throw std::runtime_error("cannot create log capture pipe");
      reader_ = descriptors[0];
      std::cout.flush(); std::cerr.flush(); std::fflush(nullptr);
      if (dup2(descriptors[1], STDOUT_FILENO) < 0 || dup2(descriptors[1], STDERR_FILENO) < 0) {
        close(descriptors[1]);
        throw std::runtime_error("cannot redirect runtime output");
      }
      close(descriptors[1]);
      capture_ = std::thread(&RuntimeSession::capture, this);
    } catch (...) {
      if (console_ >= 0) { dup2(console_, STDOUT_FILENO); close(console_); }
      if (saved_error_ >= 0) { dup2(saved_error_, STDERR_FILENO); close(saved_error_); }
      if (reader_ >= 0) close(reader_);
      unsetenv("ACTION_GO_SESSION_DIR");
      throw;
    }
    write_console("ActionGo | " + std::string(dry_run ? "DRY_RUN" : "REAL") +
                  " | " + net + "\nLogs: " + directory_.string() + "\n");
  }

  ~RuntimeSession() {
    std::cout.flush(); std::cerr.flush(); std::fflush(nullptr);
    if (console_ >= 0) dup2(console_, STDOUT_FILENO);
    if (saved_error_ >= 0) dup2(saved_error_, STDERR_FILENO);
    stopping_ = true;
    if (capture_.joinable()) capture_.join();
    if (reader_ >= 0) close(reader_);
    if (drawn_ && tty_) write_console("\n");
    if (console_ >= 0) close(console_);
    if (saved_error_ >= 0) close(saved_error_);
    nlohmann::json summary{{"clean_shutdown", std::uncaught_exceptions() == 0},
                           {"result", std::uncaught_exceptions() ? "exception" : result_},
                           {"output_logging_failed", capture_failed_.load()}};
    std::ofstream(directory_ / "summary.json") << summary.dump(2) << '\n';
    unsetenv("ACTION_GO_SESSION_DIR");
  }

  const std::string& event_path() const { return event_path_; }
  void result(const std::string& value) { result_ = value; }

  void notice(const std::string& level, const std::string& message) {
    notices_.log(level == "ERROR" ? "console_error" : "console_notice",
                 {event_string("message", message), event_string("severity", level)});
    if (drawn_ && tty_) { write_console("\033[4A\033[J"); drawn_ = false; }
    write_console(level + " " + message + "\n");
  }

  void render(const std::string& state, const std::string& policy,
              const std::string& timing_path, bool stale = false) {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_render_ < std::chrono::milliseconds(200)) return;
    last_render_ = now;
    if ((!notices_.healthy() || notices_.dropped() || capture_failed_.load()) && !logging_warning_) {
      logging_warning_ = true;
      notice("ERROR", "Logging is incomplete; check session health files");
    }
    std::string latency = "--";
    if (!policy.empty() && !timing_path.empty()) {
      try {
        const auto direct = timing_path + ".latest.json";
        std::ifstream input(std::filesystem::exists(direct) ? direct : timing_path + ".dds.jsonl.latest.json");
        const auto sample = nlohmann::json::parse(input);
        const auto stamp = sample.at("t_steady_ns").get<int64_t>();
        const auto age = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - stamp;
        if (age < 0 || age > 1000000000 || stale) latency = "STALE";
        else {
          std::ostringstream value;
          value << std::fixed << std::setprecision(1) << sample.at("e2e_ms").get<double>() << " ms";
          latency = value.str();
        }
      } catch (...) {}
    }
    const std::string displayed_state = state + (stale ? " [STALE]" : "");
    if (state != last_state_ || policy != last_policy_) {
      notices_.log("display_state_changed", {event_string("control_state", state),
                    event_string("policy", policy), event_string("source", "control_rpc_ack")});
      last_state_ = state;
      last_policy_ = policy;
    }
    if (mode_ == "quiet" || mode_ == "debug") return;
    const auto key = displayed_state + "|" + policy;
    if (!tty_) {
      if (key != last_key_ || (!policy.empty() && now - last_line_ >= std::chrono::seconds(1))) {
        write_console("STATE " + displayed_state + " | POLICY " +
                      (policy.empty() ? "--" : policy) + " | E2E " + latency + "\n");
        last_line_ = now;
      }
    } else {
      if (drawn_) write_console("\033[4A");
      write_console("\033[2K-----------------------------------------------\n"
                    "\033[2K  STATE    " + displayed_state + "\n"
                    "\033[2K  POLICY   \033[1;36m" + (policy.empty() ? "--" : policy) +
                    "\033[0m    E2E  " + latency + "\n"
                    "\033[2K-----------------------------------------------\n");
      drawn_ = true;
    }
    last_key_ = key;
  }

 private:
  void write_console(const std::string& value) {
    const char* data = value.data();
    size_t remaining = value.size();
    while (remaining) {
      const auto count = write(console_, data, remaining);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) break;
      data += count; remaining -= static_cast<size_t>(count);
    }
  }

  void capture() {
    const auto path = directory_ / "runtime.output.log";
    std::ofstream output(path, std::ios::binary);
    size_t bytes = 0;
    char buffer[8192];
    while (true) {
      pollfd descriptor{reader_, POLLIN, 0};
      const int ready = poll(&descriptor, 1, 100);
      if (ready < 0 && errno == EINTR) continue;
      if (ready <= 0) { if (stopping_) break; continue; }
      const auto count = read(reader_, buffer, sizeof(buffer));
      if (count <= 0) break;
      if (mode_ == "debug") write_console(std::string(buffer, count));
      output.write(buffer, count); output.flush();
      if (!output && !capture_failed_.exchange(true)) {
        write_console("ERROR Runtime output logging failed\n");
      }
      bytes += count;
      if (bytes >= 32 * 1024 * 1024) {
        output.close();
        std::error_code error;
        std::filesystem::remove(path.string() + ".3", error);
        for (int index = 2; index >= 0; --index) {
          const auto from = path.string() + (index ? "." + std::to_string(index) : "");
          if (std::filesystem::exists(from, error)) {
            std::filesystem::rename(from, path.string() + "." + std::to_string(index + 1), error);
          }
        }
        output.open(path, std::ios::binary | std::ios::trunc); bytes = 0;
      }
    }
  }

  std::filesystem::path directory_;
  std::string event_path_, mode_, last_key_, last_state_, last_policy_, result_{"stopped"};
  RuntimeEventLogger notices_;
  int console_{-1}, saved_error_{-1}, reader_{-1};
  bool tty_{false}, drawn_{false}, logging_warning_{false};
  std::atomic<bool> stopping_{false}, capture_failed_{false};
  std::thread capture_;
  std::chrono::steady_clock::time_point last_render_{}, last_line_{};
};

}
