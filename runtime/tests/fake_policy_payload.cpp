#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {

std::atomic<bool> g_stop{false};

void signal_handler(int) {
  g_stop.store(true);
}

void touch(const std::string& path) {
  if (path.empty()) return;
  fs::path file(path);
  if (file.has_parent_path()) fs::create_directories(file.parent_path());
  std::ofstream stream(file);
  if (!stream) throw std::runtime_error("cannot create " + path);
  stream << "ready\n";
}

void write_environment(const std::string& path) {
  if (path.empty()) return;
  fs::path file(path);
  if (file.has_parent_path()) fs::create_directories(file.parent_path());
  std::ofstream stream(file);
  if (!stream) throw std::runtime_error("cannot create " + path);
  const char* keys[] = {
      "ACTION_GO_RUN_ID",
      "ACTION_GO_TRIAL_ID",
      "ACTION_GO_PAIR_ID",
      "ACTION_GO_POLICY",
      "ACTION_GO_HANDOFF_METHOD",
      "ACTION_GO_GIT_SHA",
      "ACTION_GO_SPEC_SHA256",
      "ACTION_GO_MODEL_SHA256",
      "ACTION_GO_HARDWARE_ID",
      "ACTION_GO_MANIFEST_SHA256",
  };
  for (const char* key : keys) {
    const char* value = std::getenv(key);
    stream << key << '=' << (value ? value : "") << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  std::string ready_file;
  std::string start_file;
  std::string environment_file;
  std::string descendant_marker;
  int sleep_ms = 0;
  int run_ms = 10;
  bool no_ready = false;
  bool exit_before_ready = false;
  int exit_code = 0;
  int descendant_delay_ms = 250;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    auto need = [&]() -> std::string {
      if (index + 1 >= argc) throw std::runtime_error("missing value for " + key);
      return argv[++index];
    };
    if (key == "--ready-file") ready_file = need();
    else if (key == "--start-file") start_file = need();
    else if (key == "--environment-file") environment_file = need();
    else if (key == "--descendant-marker") descendant_marker = need();
    else if (key == "--descendant-delay-ms") descendant_delay_ms = std::stoi(need());
    else if (key == "--sleep-ms") sleep_ms = std::stoi(need());
    else if (key == "--run-ms") run_ms = std::stoi(need());
    else if (key == "--exit-code") exit_code = std::stoi(need());
    else if (key == "--no-ready") no_ready = true;
    else if (key == "--exit-before-ready") exit_before_ready = true;
  }
  if (!descendant_marker.empty()) {
    const pid_t descendant = fork();
    if (descendant < 0) return 21;
    if (descendant == 0) {
      std::signal(SIGTERM, SIG_DFL);
      std::signal(SIGINT, SIG_DFL);
      std::this_thread::sleep_for(std::chrono::milliseconds(descendant_delay_ms));
      touch(descendant_marker);
      _exit(0);
    }
  }
  if (exit_before_ready) return exit_code == 0 ? 17 : exit_code;
  if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  write_environment(environment_file);
  if (!no_ready) touch(ready_file);
  const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!g_stop.load() && !start_file.empty() && !fs::exists(start_file) &&
         std::chrono::steady_clock::now() < wait_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  const auto run_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(run_ms);
  while (!g_stop.load() && std::chrono::steady_clock::now() < run_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return g_stop.load() ? 0 : exit_code;
}
