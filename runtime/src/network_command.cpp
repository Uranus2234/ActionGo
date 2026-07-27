#include "action_go/runtime/network_command.h"

#include <nlohmann/json.hpp>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace action_go::runtime {
namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string unquote(std::string value) {
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::vector<std::string> split_words(std::string text) {
  for (char& c : text) {
    if (c == ',' || c == ';') c = ' ';
  }
  std::stringstream stream(text);
  std::vector<std::string> output;
  std::string item;
  while (stream >> item) output.push_back(unquote(item));
  return output;
}

bool parse_key_value(const std::string& token, std::string& key, std::string& value) {
  const size_t separator = token.find_first_of(":=");
  if (separator == std::string::npos || separator == 0 || separator + 1 >= token.size()) {
    return false;
  }
  key = lower(token.substr(0, separator));
  value = unquote(token.substr(separator + 1));
  return true;
}

NetworkCommandType command_type(const std::string& raw) {
  const std::string type = lower(raw);
  if (type == "action") return NetworkCommandType::kAction;
  if (type == "emergency") return NetworkCommandType::kEmergency;
  if (type == "damping_exit") return NetworkCommandType::kDampingExit;
  if (type == "walkrun_exit") return NetworkCommandType::kWalkrunExit;
  throw std::runtime_error("unsupported signed UDP command type: " + raw);
}

std::string canonical_payload(const nlohmann::json& message) {
  return std::to_string(message.at("version").get<int>()) + "\n" +
         message.at("type").get<std::string>() + "\n" +
         message.value("action", std::string()) + "\n" +
         std::to_string(message.at("issued_at_ms").get<int64_t>()) + "\n" +
         std::to_string(message.at("execute_at_ms").get<int64_t>()) + "\n" +
         message.at("nonce").get<std::string>() + "\n" +
         message.at("key_id").get<std::string>();
}

std::string hmac_sha256_hex(const std::string& key, const std::string& payload) {
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int digest_length = 0;
  if (!HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
            reinterpret_cast<const unsigned char*>(payload.data()), payload.size(),
            digest, &digest_length)) {
    throw std::runtime_error("HMAC-SHA256 failed");
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_length; ++index) {
    output << std::setw(2) << static_cast<int>(digest[index]);
  }
  return output.str();
}

bool constant_time_equal(const std::string& left, const std::string& right) {
  return left.size() == right.size() &&
         CRYPTO_memcmp(left.data(), right.data(), left.size()) == 0;
}

void require_exact_fields(const nlohmann::json& message) {
  static const std::unordered_set<std::string> allowed{
      "version", "type", "action", "issued_at_ms", "execute_at_ms",
      "nonce", "key_id", "signature"};
  for (auto it = message.begin(); it != message.end(); ++it) {
    if (allowed.find(it.key()) == allowed.end()) {
      throw std::runtime_error("signed UDP command contains unsupported field: " + it.key());
    }
  }
  for (const char* required : {"version", "type", "action", "issued_at_ms", "execute_at_ms",
                               "nonce", "key_id", "signature"}) {
    if (!message.contains(required)) {
      throw std::runtime_error(std::string("signed UDP command missing field: ") + required);
    }
  }
}

}  // namespace

NetworkCommand parse_legacy_network_command(const std::string& message) {
  const std::string raw = trim(message);
  if (raw.empty()) throw std::runtime_error("empty network command");
  const std::string lowered = lower(raw);

  NetworkCommand command;
  command.raw = raw;
  if (lowered == "emergency" || lowered == "stop") {
    command.type = NetworkCommandType::kEmergency;
    return command;
  }
  if (lowered == "damping" || lowered == "damping_exit" || lowered == "passive") {
    command.type = NetworkCommandType::kDampingExit;
    return command;
  }
  if (lowered == "exit" || lowered == "normal_exit" || lowered == "walkrun_exit" ||
      lowered == "walkrun") {
    command.type = NetworkCommandType::kWalkrunExit;
    return command;
  }

  const auto words = split_words(raw);
  if (words.empty()) throw std::runtime_error("empty network command");
  size_t begin = 0;
  const std::string prefix = lower(words.front());
  if (prefix == "action" || prefix == "run" || prefix == "start") begin = 1;

  command.type = NetworkCommandType::kAction;
  for (size_t index = begin; index < words.size(); ++index) {
    std::string key;
    std::string value;
    if (parse_key_value(words[index], key, value)) {
      if (key == "action" || key == "name") {
        command.action = value;
      } else if (key == "delay_ms" || key == "delay") {
        command.delay_ms = std::stoi(value);
      } else if (key == "start_ms" || key == "start_epoch_ms" || key == "time_ms") {
        command.execute_at_ms = std::stoll(value);
      } else {
        throw std::runtime_error("unsupported legacy UDP command field: " + key);
      }
    } else if (command.action.empty()) {
      command.action = words[index];
    } else {
      throw std::runtime_error("unexpected legacy UDP command token: " + words[index]);
    }
  }
  if (command.action.empty()) {
    throw std::runtime_error("network action command has no action name: " + raw);
  }
  if (command.delay_ms < 0) throw std::runtime_error("network delay_ms must be non-negative");
  return command;
}

bool is_priority_network_command(NetworkCommandType type) {
  return type != NetworkCommandType::kAction;
}

bool is_loopback_bind_address(const std::string& address) {
  const std::string value = lower(trim(address));
  if (value.empty() || value == "*" || value == "0.0.0.0" || value == "::" ||
      value == "[::]" || value == "255.255.255.255") {
    return false;
  }
  if (value == "localhost" || value == "127.0.0.1" || value == "::1" ||
      value == "[::1]") {
    return true;
  }
  in_addr v4{};
  if (::inet_pton(AF_INET, value.c_str(), &v4) == 1) {
    return (ntohl(v4.s_addr) >> 24) == 127;
  }
  in6_addr v6{};
  if (::inet_pton(AF_INET6, value.c_str(), &v6) == 1) {
    return IN6_IS_ADDR_LOOPBACK(&v6);
  }
  return false;
}

std::string load_udp_auth_key_file(const std::string& path) {
  if (path.empty()) throw std::runtime_error("UDP authentication key path is empty");
  namespace fs = std::filesystem;
  const fs::path key_path(path);
  if (!fs::is_regular_file(key_path)) {
    throw std::runtime_error("UDP authentication key file not found: " + path);
  }
  const auto permissions = fs::status(key_path).permissions();
  using fs::perms;
  if ((permissions & (perms::group_all | perms::others_all)) != perms::none) {
    throw std::runtime_error("UDP authentication key file must not be accessible by group or other users: " + path);
  }
  std::ifstream input(key_path, std::ios::binary);
  std::ostringstream contents;
  contents << input.rdbuf();
  std::string key = trim(contents.str());
  if (key.size() < 32) {
    throw std::runtime_error("UDP authentication key must contain at least 32 bytes");
  }
  return key;
}

BoundedNetworkCommandQueue::BoundedNetworkCommandQueue(size_t capacity)
    : capacity_(capacity) {
  if (capacity_ == 0) throw std::runtime_error("network command queue capacity must be positive");
}

bool BoundedNetworkCommandQueue::push(NetworkCommand command) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto rank = [](NetworkCommandType type) {
    switch (type) {
      case NetworkCommandType::kEmergency: return 0;
      case NetworkCommandType::kDampingExit: return 1;
      case NetworkCommandType::kWalkrunExit: return 2;
      case NetworkCommandType::kAction: return 3;
    }
    return 3;
  };
  if (command.type == NetworkCommandType::kEmergency) {
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [](const NetworkCommand& queued) {
      return queued.type != NetworkCommandType::kEmergency;
    }), queue_.end());
    queue_.push_front(std::move(command));
    return true;
  }
  if (queue_.size() >= capacity_) {
    if (is_priority_network_command(command.type)) {
      auto removable = std::find_if(queue_.rbegin(), queue_.rend(), [&](const NetworkCommand& queued) {
        return rank(queued.type) > rank(command.type);
      });
      if (removable == queue_.rend()) return false;
      queue_.erase(std::next(removable).base());
    } else {
      return false;
    }
  }
  auto insertion = std::find_if(queue_.begin(), queue_.end(), [&](const NetworkCommand& queued) {
    return rank(command.type) < rank(queued.type);
  });
  queue_.insert(insertion, std::move(command));
  return true;
}

bool BoundedNetworkCommandQueue::pop(NetworkCommand& command) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) return false;
  command = std::move(queue_.front());
  queue_.pop_front();
  return true;
}

size_t BoundedNetworkCommandQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

AuthenticatedNetworkCommandParser::AuthenticatedNetworkCommandParser(Config config)
    : config_(std::move(config)) {
  if (config_.replay_capacity == 0) throw std::runtime_error("UDP replay capacity must be positive");
  if (config_.max_age_ms <= 0 || config_.max_future_ms <= 0) {
    throw std::runtime_error("UDP authentication time windows must be positive");
  }
}

void AuthenticatedNetworkCommandParser::expire_nonces(int64_t now_epoch_ms) {
  while (!nonce_order_.empty()) {
    const auto found = nonce_expiry_.find(nonce_order_.front());
    if (found != nonce_expiry_.end() && found->second > now_epoch_ms) {
      break;
    }
    if (found != nonce_expiry_.end()) nonce_expiry_.erase(found);
    nonce_order_.pop_front();
  }
}

NetworkCommand AuthenticatedNetworkCommandParser::parse(
    const std::string& message, int64_t now_epoch_ms, bool require_authentication) {
  const std::string raw = trim(message);
  if (!require_authentication && (raw.empty() || raw.front() != '{')) {
    return parse_legacy_network_command(raw);
  }
  if (config_.key.empty()) {
    throw std::runtime_error("signed UDP command received without a configured authentication key");
  }

  const nlohmann::json json = nlohmann::json::parse(raw);
  if (!json.is_object()) throw std::runtime_error("signed UDP command must be a JSON object");
  require_exact_fields(json);
  if (json.at("version").get<int>() != 1) {
    throw std::runtime_error("unsupported signed UDP command version");
  }
  const std::string nonce = json.at("nonce").get<std::string>();
  if (nonce.size() < 16 || nonce.size() > 128 ||
      !std::all_of(nonce.begin(), nonce.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_';
      })) {
    throw std::runtime_error("signed UDP command nonce must be 16-128 URL-safe characters");
  }
  if (!config_.key_id.empty() && json.at("key_id").get<std::string>() != config_.key_id) {
    throw std::runtime_error("signed UDP command key_id does not match configured key");
  }
  const int64_t issued_at_ms = json.at("issued_at_ms").get<int64_t>();
  const int64_t execute_at_ms = json.at("execute_at_ms").get<int64_t>();
  if (issued_at_ms < now_epoch_ms - config_.max_age_ms) {
    throw std::runtime_error("signed UDP command has expired");
  }
  if (issued_at_ms > now_epoch_ms + config_.max_future_ms ||
      execute_at_ms > now_epoch_ms + config_.max_future_ms) {
    throw std::runtime_error("signed UDP command is too far in the future");
  }
  if (execute_at_ms < issued_at_ms) {
    throw std::runtime_error("signed UDP command execute_at_ms precedes issued_at_ms");
  }

  const std::string supplied_signature = lower(json.at("signature").get<std::string>());
  const std::string expected_signature = hmac_sha256_hex(config_.key, canonical_payload(json));
  if (!constant_time_equal(supplied_signature, expected_signature)) {
    throw std::runtime_error("signed UDP command has an invalid signature");
  }

  NetworkCommand command;
  command.type = command_type(json.at("type").get<std::string>());
  command.action = json.at("action").get<std::string>();
  command.execute_at_ms = execute_at_ms;
  command.raw = raw;
  if (command.type == NetworkCommandType::kAction && command.action.empty()) {
    throw std::runtime_error("signed action command requires a non-empty action");
  }
  if (command.type != NetworkCommandType::kAction && !command.action.empty()) {
    throw std::runtime_error("non-action signed command must not specify an action");
  }

  expire_nonces(now_epoch_ms);
  if (nonce_expiry_.find(nonce) != nonce_expiry_.end()) {
    throw std::runtime_error("signed UDP command nonce was already used");
  }
  if (nonce_expiry_.size() >= config_.replay_capacity) {
    throw std::runtime_error("signed UDP replay cache is full");
  }
  nonce_expiry_[nonce] = now_epoch_ms + config_.max_age_ms + config_.max_future_ms;
  nonce_order_.push_back(nonce);
  return command;
}

}  // namespace action_go::runtime
