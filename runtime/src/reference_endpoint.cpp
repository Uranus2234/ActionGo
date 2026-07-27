#include "action_go/runtime/reference_endpoint.h"

#include <arpa/inet.h>

namespace action_go::runtime {
namespace {

bool is_loopback_host(const std::string& host) {
  if (host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]") {
    return true;
  }
  in_addr v4{};
  if (::inet_pton(AF_INET, host.c_str(), &v4) == 1) {
    return (ntohl(v4.s_addr) >> 24) == 127;
  }
  in6_addr v6{};
  return ::inet_pton(AF_INET6, host.c_str(), &v6) == 1 && IN6_IS_ADDR_LOOPBACK(&v6);
}

}  // namespace

bool is_local_reference_bind(const std::string& address) {
  return is_loopback_host(address);
}

bool is_local_reference_endpoint(const std::string& endpoint) {
  if (endpoint.rfind("ipc://", 0) == 0 || endpoint.rfind("inproc://", 0) == 0) {
    return true;
  }
  if (endpoint.rfind("tcp://", 0) != 0) return false;

  const std::string host_port = endpoint.substr(6);
  if (host_port.empty()) return false;
  std::string host;
  if (host_port.front() == '[') {
    const size_t close = host_port.find(']');
    if (close == std::string::npos || close == 1) return false;
    host = host_port.substr(1, close - 1);
    if (close + 1 < host_port.size() && host_port[close + 1] != ':') return false;
  } else {
    const size_t colon = host_port.rfind(':');
    if (colon == std::string::npos || colon == 0) return false;
    host = host_port.substr(0, colon);
    if (host.find(':') != std::string::npos) return false;
  }
  return is_loopback_host(host);
}

}  // namespace action_go::runtime
