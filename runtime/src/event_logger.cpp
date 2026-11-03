#include "action_go/runtime/event_logger.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <unistd.h>

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

RuntimeEventLogger::~RuntimeEventLogger() { stop(); }

void RuntimeEventLogger::stop() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stopping_ = true;
  }
  condition_.notify_one();
  if (writer_.joinable()) writer_.join();
}

void RuntimeEventLogger::start(const std::string& path, bool append) {
  stop();
  std::lock_guard<std::mutex> lock(mtx_);
  path_ = path;
  t0_ = std::chrono::steady_clock::now();
  if (path_.empty()) return;
  fs::path p(path_);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream probe(path_, std::ios::out | (append ? std::ios::app : std::ios::trunc));
  if (!probe) throw std::runtime_error("cannot open event log: " + path_);
  stopping_ = false;
  append_ = append;
  dropped_ = 0;
  failed_ = false;
  overloaded_ = false;
  writer_ = std::thread(&RuntimeEventLogger::write_loop, this);
}

void RuntimeEventLogger::open(const std::string& path) { start(path, false); }
void RuntimeEventLogger::open_append(const std::string& path) { start(path, true); }

bool RuntimeEventLogger::enabled() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return !path_.empty();
}

void RuntimeEventLogger::flush() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (path_.empty() || !writer_.joinable() || failed_) return;
  auto barrier = std::make_shared<std::promise<void>>();
  auto complete = barrier->get_future();
  {
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    if (failed_) return;
    queue_.push_back({{}, {}, false, false, barrier});
  }
  condition_.notify_one();
  complete.wait();
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
  std::ostringstream out;
  const bool critical = event.find("fail") != std::string::npos ||
      event.find("error") != std::string::npos || event.find("timeout") != std::string::npos ||
      event == "safety_stop" || event == "switch_back_blocked" ||
      std::any_of(fields.begin(), fields.end(), [](const auto& field) {
        return field.key == "ok" && field.value == "false";
      });
  const std::string level = critical ? "ERROR" :
      (event == "command_timing" || event == "runtime_metrics" ? "DEBUG" : "INFO");
  out << "{\"event\":\"" << json_escape(event) << "\",\"t_mono_sec\":"
      << std::fixed << std::setprecision(9) << t
      << ",\"t_steady_ns\":" << steady_ns
      << ",\"t_unix_ns\":" << unix_ns
      << ",\"level\":\"" << level << "\",\"pid\":" << getpid()
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
  std::ostringstream text;
  const auto seconds = std::chrono::system_clock::to_time_t(unix_now);
  std::tm utc{};
  gmtime_r(&seconds, &utc);
  text << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
       << std::setfill('0') << std::setw(3) << (unix_ns / 1000000 % 1000)
       << "Z " << level << " pid=" << getpid() << " policy=" << context_.policy << ' ' << event;
  for (const auto& field : fields) text << ' ' << field.key << '=' << field.value;
  text << '\n';
  {
    std::unique_lock<std::mutex> queue_lock(queue_mutex_, std::try_to_lock);
    if (!queue_lock || failed_ || stopping_ || queue_.size() >= (critical ? 1152u : 1024u)) {
      ++dropped_;
      overloaded_ = true;
      return;
    }
    queue_.push_back({out.str(), text.str(), critical, event == "command_timing"});
  }
  condition_.notify_one();
}

void RuntimeEventLogger::write_loop() {
  try {
    const auto mode = std::ios::out | (append_ ? std::ios::app : std::ios::trunc);
    const auto text_path = fs::path(path_).extension() == ".jsonl"
        ? fs::path(path_).replace_extension(".log").string() : path_ + ".log";
    std::ofstream json(path_, mode), text(text_path, mode);
    if (!json || !text) throw std::runtime_error("cannot open log streams");
    uintmax_t bytes = fs::file_size(path_);
    uintmax_t limit = 32 * 1024 * 1024;
    if (const auto configured = environment_value("ACTION_GO_LOG_MAX_BYTES"); !configured.empty()) {
      limit = std::max<uintmax_t>(4096, std::stoull(configured));
    }
    auto last_flush = std::chrono::steady_clock::now();
    while (true) {
      Record record;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(200), [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          json.flush(); text.flush();
          if (!json || !text) throw std::runtime_error("log flush failed");
          if (stopping_) break;
          continue;
        }
        record = std::move(queue_.front());
        queue_.pop_front();
      }
      if (record.barrier) {
        json.flush(); text.flush();
        record.barrier->set_value();
        if (!json || !text) throw std::runtime_error("log flush failed");
        continue;
      }
      if (overloaded_.exchange(false)) {
        std::fprintf(stderr, "[logging] records dropped from %s; count=%llu\n",
                     path_.c_str(), dropped_.load());
      }
      if (bytes >= limit) {
        json.close(); text.close();
        for (const auto& base : {path_, text_path}) {
          std::error_code error;
          fs::remove(base + ".3", error);
          for (int index = 2; index >= 0; --index) {
            const auto from = base + (index ? "." + std::to_string(index) : "");
            if (fs::exists(from)) fs::rename(from, base + "." + std::to_string(index + 1));
          }
        }
        json.open(path_, std::ios::trunc); text.open(text_path, std::ios::trunc);
        bytes = 0;
      }
      json << record.json; text << record.text;
      bytes += record.json.size();
      if (record.timing) {
        const auto temporary = path_ + ".latest.tmp";
        std::ofstream latest(temporary, std::ios::trunc);
        latest << record.json;
        latest.close();
        if (!latest) throw std::runtime_error("cannot write timing status");
        fs::rename(temporary, path_ + ".latest.json");
      }
      const auto now = std::chrono::steady_clock::now();
      if (record.critical || now - last_flush >= std::chrono::milliseconds(200)) {
        json.flush(); text.flush(); last_flush = now;
      }
      if (!json || !text) throw std::runtime_error("log write failed");
    }
  } catch (const std::exception& error) {
    failed_ = true;
    std::lock_guard<std::mutex> lock(queue_mutex_);
    for (auto& record : queue_) if (record.barrier) record.barrier->set_value();
    dropped_ += queue_.size();
    queue_.clear();
    std::fprintf(stderr, "[logging] %s: %s\n", path_.c_str(), error.what());
  }
  std::ofstream health(path_ + ".health.json");
  health << "{\"failed\":" << (failed_ ? "true" : "false")
         << ",\"dropped_records\":" << dropped_.load() << "}\n";
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
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          const char digits[] = "0123456789abcdef";
          out += "\\u00";
          out += digits[(static_cast<unsigned char>(c) >> 4) & 15];
          out += digits[static_cast<unsigned char>(c) & 15];
        } else out += c;
        break;
    }
  }
  return out;
}

}  // namespace action_go::runtime
