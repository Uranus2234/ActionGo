#pragma once

#include "action_go/runtime/signal_file.h"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace action_go::runtime {

class InputLoss final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline bool request_input_loss_walkrun(const std::string& reason) {
  const char* request = std::getenv("ACTION_GO_INPUT_LOSS_FILE");
  const char* release = std::getenv("ACTION_GO_INPUT_LOSS_RELEASE");
  if (!request || !*request || !release || !*release) return false;
  write_signal_file(request, reason + "\n", true);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (signal_file_exists(release)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  throw std::runtime_error("WALKRUN handoff was not acknowledged: " + reason);
}

class TwoFrameInputGate {
 public:
  bool check(bool valid, const std::string& reason) {
    if (valid) {
      failures_ = 0;
      return true;
    }
    if (++failures_ >= 2) throw InputLoss(reason);
    return false;
  }

 private:
  unsigned failures_{0};
};

}
