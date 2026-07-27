#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <sys/types.h>

namespace action_go::runtime {

enum class SupervisorProcessState {
  kRunning,
  kExited,
  kError,
};

struct SupervisorProcessResult {
  SupervisorProcessState state{SupervisorProcessState::kRunning};
  int exit_code{0};
  int signal{0};
  int error_number{0};
  std::string error;
};

class SupervisorPlatform {
 public:
  using TimePoint = std::chrono::steady_clock::time_point;

  virtual ~SupervisorPlatform() = default;

  virtual TimePoint now() = 0;
  virtual void sleep_for(std::chrono::milliseconds duration) = 0;
  virtual bool file_exists(const std::string& path) = 0;
  virtual bool remove_file(const std::string& path, std::string& error) = 0;
  virtual bool touch_file(const std::string& path, const std::string& contents,
                          std::string& error) = 0;
  virtual bool start_shell(const std::string& command, pid_t& pid,
                           std::string& error) = 0;
  virtual SupervisorProcessResult poll_process(pid_t pid) = 0;
  virtual bool terminate_process_group(pid_t pid, bool force,
                                       std::string& error) = 0;
};

std::shared_ptr<SupervisorPlatform> make_posix_supervisor_platform();

}  // namespace action_go::runtime
