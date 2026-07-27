#include "action_go/runtime/event_logger.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cmath>

namespace fs = std::filesystem;

namespace action_go::runtime {

namespace {

std::string environment_value(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string{};
}

}  // namespace

ActionLifecycleEventV1 action_lifecycle_context_from_environment() {
  ActionLifecycleEventV1 context;
  context.run_id = environment_value("ACTION_GO_RUN_ID");
  context.trial_id = environment_value("ACTION_GO_TRIAL_ID");
  context.pair_id = environment_value("ACTION_GO_PAIR_ID");
  context.policy = environment_value("ACTION_GO_POLICY");
  context.handoff_method = environment_value("ACTION_GO_HANDOFF_METHOD");
  context.git_sha = environment_value("ACTION_GO_GIT_SHA");
  context.spec_sha256 = environment_value("ACTION_GO_SPEC_SHA256");
  context.model_sha256 = environment_value("ACTION_GO_MODEL_SHA256");
  context.hardware_id = environment_value("ACTION_GO_HARDWARE_ID");
  context.manifest_sha256 = environment_value("ACTION_GO_MANIFEST_SHA256");
  return context;
}

RuntimeEventField event_string(std::string key, std::string value) {
  return {std::move(key), std::move(value), true};
}

RuntimeEventField event_number(std::string key, double value) {
  if (!std::isfinite(value)) {
    return {std::move(key), "null", false};
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(9) << value;
  return {std::move(key), out.str(), false};
}

RuntimeEventField event_int(std::string key, long long value) {
  return {std::move(key), std::to_string(value), false};
}

RuntimeEventField event_bool(std::string key, bool value) {
  return {std::move(key), value ? "true" : "false", false};
}

RuntimeEventLogger::RuntimeEventLogger()
    : t0_(std::chrono::steady_clock::now()),
      context_(action_lifecycle_context_from_environment()) {}

void RuntimeEventLogger::open(const std::string& path) {
  std::lock_guard<std::mutex> lock(mtx_);
  path_ = path;
  t0_ = std::chrono::steady_clock::now();
  if (path_.empty()) return;
  fs::path p(path_);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream out(path_, std::ios::out | std::ios::trunc);
}

void RuntimeEventLogger::open_append(const std::string& path) {
  std::lock_guard<std::mutex> lock(mtx_);
  path_ = path;
  t0_ = std::chrono::steady_clock::now();
  if (path_.empty()) return;
  fs::path p(path_);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream out(path_, std::ios::out | std::ios::app);
}

bool RuntimeEventLogger::enabled() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return !path_.empty();
}

void RuntimeEventLogger::set_context(ActionLifecycleEventV1 context) {
  std::lock_guard<std::mutex> lock(mtx_);
  context_ = std::move(context);
}

ActionLifecycleEventV1 RuntimeEventLogger::context() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return context_;
}

void RuntimeEventLogger::log(const std::string& event,
                             std::initializer_list<RuntimeEventField> fields) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (path_.empty()) return;
  const auto now = std::chrono::steady_clock::now();
  const auto unix_now = std::chrono::system_clock::now();
  const double t = std::chrono::duration<double>(now - t0_).count();
  const auto steady_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now.time_since_epoch()).count();
  const auto unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      unix_now.time_since_epoch()).count();
  std::ofstream out(path_, std::ios::out | std::ios::app);
  out << "{\"event\":\"" << json_escape(event) << "\",\"t_mono_sec\":"
      << std::fixed << std::setprecision(9) << t
      << ",\"t_steady_ns\":" << steady_ns
      << ",\"t_unix_ns\":" << unix_ns
      << ",\"schema_version\":" << context_.schema_version
      << ",\"run_id\":\"" << json_escape(context_.run_id) << "\""
      << ",\"trial_id\":\"" << json_escape(context_.trial_id) << "\""
      << ",\"pair_id\":\"" << json_escape(context_.pair_id) << "\""
      << ",\"policy\":\"" << json_escape(context_.policy) << "\""
      << ",\"handoff_method\":\"" << json_escape(context_.handoff_method) << "\""
      << ",\"git_sha\":\"" << json_escape(context_.git_sha) << "\""
      << ",\"spec_sha256\":\"" << json_escape(context_.spec_sha256) << "\""
      << ",\"model_sha256\":\"" << json_escape(context_.model_sha256) << "\""
      << ",\"hardware_id\":\"" << json_escape(context_.hardware_id) << "\""
      << ",\"manifest_sha256\":\"" << json_escape(context_.manifest_sha256) << "\"";
  for (const RuntimeEventField& field : fields) {
    out << ",\"" << json_escape(field.key) << "\":";
    if (field.quote) {
      out << "\"" << json_escape(field.value) << "\"";
    } else {
      out << field.value;
    }
  }
  out << "}\n";
}

std::string RuntimeEventLogger::json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

}  // namespace action_go::runtime
