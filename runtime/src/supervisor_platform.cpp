#include "action_go/runtime/supervisor_platform.h"

#include <cerrno>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

namespace fs = std::filesystem;

namespace action_go::runtime {
namespace {

SupervisorProcessResult decode_process_status(int status) {
  SupervisorProcessResult result;
  result.state = SupervisorProcessState::kExited;
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.signal = WTERMSIG(status);
  } else {
    result.state = SupervisorProcessState::kError;
    result.error = "unexpected child process status=" + std::to_string(status);
  }
  return result;
}

class PosixSupervisorPlatform final : public SupervisorPlatform {
 public:
  TimePoint now() override {
    return std::chrono::steady_clock::now();
  }

  void sleep_for(std::chrono::milliseconds duration) override {
    std::this_thread::sleep_for(duration);
  }

  bool file_exists(const std::string& path) override {
    std::error_code error;
    const bool exists = fs::exists(path, error);
    return !error && exists;
  }

  bool remove_file(const std::string& path, std::string& error) override {
    if (path.empty()) return true;
    std::error_code filesystem_error;
    fs::remove(path, filesystem_error);
    if (filesystem_error) {
      error = "failed to remove file " + path + ": " + filesystem_error.message();
      return false;
    }
    return true;
  }

  bool touch_file(const std::string& path, const std::string& contents,
                  std::string& error) override {
    if (path.empty()) return true;
    try {
      const fs::path file(path);
      if (file.has_parent_path()) fs::create_directories(file.parent_path());
      std::ofstream output(path);
      if (!output) {
        error = "failed to create file: " + path;
        return false;
      }
      output << contents;
      return output.good();
    } catch (const std::exception& exception) {
      error = "failed to touch file " + path + ": " + exception.what();
      return false;
    }
  }

  bool start_shell(const std::string& command, pid_t& pid,
                   std::string& error) override {
    const pid_t parent_pid = getpid();
    pid = fork();
    if (pid < 0) {
      error = "fork failed errno=" + std::to_string(errno);
      return false;
    }
    if (pid == 0) {
      setpgid(0, 0);
#ifdef __linux__
      // Ensure a payload cannot outlive the supervisor after SIGKILL/OOM.
      if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) _exit(126);
      if (getppid() != parent_pid) {
        // The parent may have exited between fork and prctl.
        static_cast<void>(kill(getpid(), SIGTERM));
      }
#endif
      execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char*>(nullptr));
      _exit(127);
    }
    setpgid(pid, pid);
    return true;
  }

  SupervisorProcessResult poll_process(pid_t pid) override {
    while (true) {
      int status = 0;
      const pid_t result = waitpid(pid, &status, WNOHANG);
      if (result == pid) return decode_process_status(status);
      if (result == 0) return {};
      if (errno == EINTR) continue;
      SupervisorProcessResult failure;
      failure.state = SupervisorProcessState::kError;
      failure.error_number = errno;
      failure.error = "waitpid failed errno=" + std::to_string(errno);
      return failure;
    }
  }

  bool terminate_process_group(pid_t pid, bool force,
                               std::string& error) override {
    if (pid <= 0) return true;
    const int initial_signal = force ? SIGKILL : SIGTERM;
    if (kill(-pid, initial_signal) != 0 && kill(pid, initial_signal) != 0 &&
        errno != ESRCH) {
      error = "failed to signal process group errno=" + std::to_string(errno);
      return false;
    }

    if (!force) {
      for (int attempt = 0; attempt < 20; ++attempt) {
        const SupervisorProcessResult result = poll_process(pid);
        if (result.state == SupervisorProcessState::kExited) return true;
        if (result.state == SupervisorProcessState::kError) {
          if (result.error_number == ECHILD) return true;
          error = result.error;
          return false;
        }
        sleep_for(std::chrono::milliseconds(20));
      }
      if (kill(-pid, SIGKILL) != 0 && kill(pid, SIGKILL) != 0 && errno != ESRCH) {
        error = "failed to kill process group errno=" + std::to_string(errno);
        return false;
      }
    }

    while (true) {
      int status = 0;
      const pid_t result = waitpid(pid, &status, 0);
      if (result == pid || (result < 0 && errno == ECHILD)) return true;
      if (result < 0 && errno == EINTR) continue;
      error = "failed to reap process errno=" + std::to_string(errno);
      return false;
    }
  }
};

}  // namespace

std::shared_ptr<SupervisorPlatform> make_posix_supervisor_platform() {
  return std::make_shared<PosixSupervisorPlatform>();
}

}  // namespace action_go::runtime
