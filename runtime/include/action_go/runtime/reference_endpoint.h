#pragma once

#include <string>

namespace action_go::runtime {

// Returns true only for local IPC/inproc endpoints or TCP endpoints whose host
// resolves syntactically to loopback. DNS names are intentionally rejected.
bool is_local_reference_bind(const std::string& address);
bool is_local_reference_endpoint(const std::string& endpoint);

}  // namespace action_go::runtime
