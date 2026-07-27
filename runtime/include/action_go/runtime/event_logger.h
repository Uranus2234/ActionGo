#pragma once

#include <chrono>
#include <initializer_list>
#include <mutex>
#include <string>
#include <utility>

namespace action_go::runtime {

inline constexpr int kActionLifecycleEventSchemaVersion = 1;

// Stable metadata attached to every supervisor, player, and trace event.
// Experiment runners populate these fields through ACTION_GO_* environment
// variables so independent processes remain joinable without sharing memory.
struct ActionLifecycleEventV1 {
  int schema_version{kActionLifecycleEventSchemaVersion};
  std::string run_id;
  std::string trial_id;
  std::string pair_id;
  std::string policy;
  std::string handoff_method;
  std::string git_sha;
  std::string spec_sha256;
  std::string model_sha256;
  std::string hardware_id;
  std::string manifest_sha256;
};

ActionLifecycleEventV1 action_lifecycle_context_from_environment();

struct RuntimeEventField {
  std::string key;
  std::string value;
  bool quote{true};
};

RuntimeEventField event_string(std::string key, std::string value);
RuntimeEventField event_number(std::string key, double value);
RuntimeEventField event_int(std::string key, long long value);
RuntimeEventField event_bool(std::string key, bool value);

class RuntimeEventLogger {
 public:
  RuntimeEventLogger();

  void open(const std::string& path);
  void open_append(const std::string& path);
  bool enabled() const;
  void set_context(ActionLifecycleEventV1 context);
  ActionLifecycleEventV1 context() const;

  void log(const std::string& event,
           std::initializer_list<RuntimeEventField> fields = {});

 private:
  static std::string json_escape(const std::string& value);

  std::string path_;
  std::chrono::steady_clock::time_point t0_;
  ActionLifecycleEventV1 context_;
  mutable std::mutex mtx_;
};

}  // namespace action_go::runtime
