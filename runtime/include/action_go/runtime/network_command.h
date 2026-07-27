#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace action_go::runtime {

enum class NetworkCommandType {
  kAction,
  kEmergency,
  kDampingExit,
  kWalkrunExit,
};

struct NetworkCommand {
  NetworkCommandType type{NetworkCommandType::kAction};
  std::string action;
  int delay_ms{0};
  int64_t execute_at_ms{0};
  std::string raw;
};

NetworkCommand parse_legacy_network_command(const std::string& message);
bool is_priority_network_command(NetworkCommandType type);
bool is_loopback_bind_address(const std::string& address);
std::string load_udp_auth_key_file(const std::string& path);

class BoundedNetworkCommandQueue {
 public:
  explicit BoundedNetworkCommandQueue(size_t capacity = 64);
  bool push(NetworkCommand command);
  bool pop(NetworkCommand& command);
  size_t size() const;

 private:
  const size_t capacity_;
  mutable std::mutex mutex_;
  std::deque<NetworkCommand> queue_;
};

class AuthenticatedNetworkCommandParser {
 public:
  struct Config {
    std::string key;
    std::string key_id;
    int64_t max_age_ms{10000};
    int64_t max_future_ms{60000};
    size_t replay_capacity{4096};
  };

  explicit AuthenticatedNetworkCommandParser(Config config);

  NetworkCommand parse(const std::string& message, int64_t now_epoch_ms,
                       bool require_authentication);

 private:
  void expire_nonces(int64_t now_epoch_ms);

  Config config_;
  std::unordered_map<std::string, int64_t> nonce_expiry_;
  std::deque<std::string> nonce_order_;
};

}  // namespace action_go::runtime
