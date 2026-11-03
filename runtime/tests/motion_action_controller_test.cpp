#include "action_go/motion_action_controller.h"
#include "action_go/runtime/control_mode_backend.h"
#include "action_go/runtime/signal_file.h"
#include <nlohmann/json.hpp>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <unistd.h>

namespace fs = std::filesystem;
using action_go::runtime::ControlModeBackend;
using action_go::runtime::OfficialControlMode;
using action_go::runtime::SupervisorPlatform;
using action_go::runtime::SupervisorProcessResult;
using action_go::runtime::SupervisorProcessState;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "[motion_action_controller_test] FAIL " << message << '\n';
  }
}

class FakeControlModeBackend final : public ControlModeBackend {
 public:
  const char* name() const override { return "fake"; }

  bool stop_move(std::string& error) override {
    ++stop_calls;
    if (fail_stop) {
      error = "injected StopMove failure";
      return false;
    }
    return true;
  }

  bool switch_to_user_ctrl(bool publish_prearm, std::string& error) override {
    ++user_calls;
    last_prearm = publish_prearm;
    if (user_delay_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(user_delay_ms));
    }
    if (fail_user) {
      error = "injected UserCtrl failure";
      return false;
    }
    active_user_ctrl.store(true);
    return true;
  }

  bool switch_to_internal(OfficialControlMode mode, std::string& error) override {
    ++internal_calls;
    last_mode = mode;
    if (internal_delay_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(internal_delay_ms));
    }
    if (fail_internal) {
      error = "injected internal-control failure";
      return false;
    }
    active_user_ctrl.store(false);
    return true;
  }

  bool enter_damping(std::string& error) override {
    ++damping_calls;
    if (fail_damping) {
      error = "injected damping failure";
      return false;
    }
    return true;
  }

  bool fail_stop{false};
  bool fail_user{false};
  bool fail_internal{false};
  bool fail_damping{false};
  int user_delay_ms{0};
  int internal_delay_ms{0};
  std::atomic<int> stop_calls{0};
  std::atomic<int> user_calls{0};
  std::atomic<int> internal_calls{0};
  std::atomic<int> damping_calls{0};
  std::atomic<bool> last_prearm{false};
  std::atomic<bool> active_user_ctrl{false};
  OfficialControlMode last_mode{OfficialControlMode::kLast};
};

class FakeSupervisorPlatform final : public SupervisorPlatform {
 public:
  TimePoint now() override {
    std::lock_guard<std::mutex> lock(mutex);
    return current_time;
  }

  void sleep_for(std::chrono::milliseconds duration) override {
    if (block_long_sleep && duration >= std::chrono::milliseconds(300)) {
      long_sleep_entered.store(true);
      long_sleep_duration_ms.store(duration.count());
      while (!allow_long_sleep.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    std::lock_guard<std::mutex> lock(mutex);
    current_time += duration;
  }

  bool file_exists(const std::string& path) override {
    std::lock_guard<std::mutex> lock(mutex);
    if (provide_ready && running && path == ready_file) return true;
    return files.count(path) != 0;
  }

  bool remove_file(const std::string& path, std::string& error) override {
    std::lock_guard<std::mutex> lock(mutex);
    ++remove_calls;
    removed_files.insert(path);
    if (fail_remove) {
      error = "injected remove failure";
      return false;
    }
    files.erase(path);
    return true;
  }

  bool touch_file(const std::string& path, const std::string&,
                  std::string& error) override {
    std::lock_guard<std::mutex> lock(mutex);
    ++touch_calls;
    if (fail_touch) {
      error = "injected touch failure";
      return false;
    }
    files.insert(path);
    if (path == start_file) start_touched = true;
    if (path == release_file) release_touched = true;
    return true;
  }

  bool start_shell(const std::string& command, pid_t& pid,
                   std::string& error) override {
    std::lock_guard<std::mutex> lock(mutex);
    ++spawn_calls;
    last_command = command;
    if (fail_spawn) {
      error = "injected spawn failure";
      return false;
    }
    pid = next_pid++;
    running = true;
    start_touched = false;
    return true;
  }

  SupervisorProcessResult poll_process(pid_t) override {
    std::lock_guard<std::mutex> lock(mutex);
    ++poll_calls;
    if (exit_before_ready) {
      running = false;
      SupervisorProcessResult result;
      result.state = SupervisorProcessState::kExited;
      result.exit_code = 19;
      return result;
    }
    if (running && complete_after_start && start_touched) {
      running = false;
      SupervisorProcessResult result;
      result.state = SupervisorProcessState::kExited;
      result.exit_code = completion_exit_code;
      return result;
    }
    if (running && hold_after_start && start_touched) {
      files.insert(complete_file);
      if (release_touched) {
        running = false;
        SupervisorProcessResult result;
        result.state = SupervisorProcessState::kExited;
        result.exit_code = completion_exit_code;
        return result;
      }
    }
    return {};
  }

  bool terminate_process_group(pid_t, bool force,
                               std::string&) override {
    std::lock_guard<std::mutex> lock(mutex);
    ++terminate_calls;
    terminate_time = current_time;
    last_terminate_forced = force;
    running = false;
    return true;
  }

  std::mutex mutex;
  TimePoint current_time{};
  std::unordered_set<std::string> files;
  std::unordered_set<std::string> removed_files;
  std::string ready_file;
  std::string start_file;
  std::string complete_file;
  std::string release_file;
  std::string last_command;
  bool provide_ready{true};
  bool fail_remove{false};
  bool fail_touch{false};
  bool fail_spawn{false};
  bool exit_before_ready{false};
  bool complete_after_start{true};
  bool hold_after_start{false};
  bool start_touched{false};
  bool release_touched{false};
  bool running{false};
  bool last_terminate_forced{false};
  bool block_long_sleep{false};
  std::atomic<bool> long_sleep_entered{false};
  std::atomic<bool> allow_long_sleep{false};
  std::atomic<int64_t> long_sleep_duration_ms{0};
  int remove_calls{0};
  int touch_calls{0};
  int spawn_calls{0};
  int poll_calls{0};
  int terminate_calls{0};
  int completion_exit_code{0};
  TimePoint terminate_time{};
  pid_t next_pid{10000};
};

struct TestFiles {
  fs::path root;
  fs::path config;
  fs::path log;
  fs::path environment;
  fs::path ready;
  fs::path start;
  fs::path complete;
  fs::path release;
  fs::path descendant_marker;

  explicit TestFiles(const std::string& name) {
    root = fs::temp_directory_path() /
        ("action_go_controller_" + name + "_" + std::to_string(::getpid()));
    fs::create_directories(root);
    config = root / "motion.yaml";
    log = root / "events.jsonl";
    environment = root / "payload_environment.txt";
    ready = root / "ready";
    start = root / "start";
    complete = root / "complete";
    release = root / "release";
    descendant_marker = root / "descendant_survived";
  }

  ~TestFiles() {
    std::error_code error;
    fs::remove_all(root, error);
  }
};

void write_config(const TestFiles& files, const std::string& action,
                  const std::string& payload_args, double ready_timeout = 0.5,
                  bool switch_back = true, bool postrun_hold = false,
                  double switch_back_overlap_sec = 0.30) {
  std::ofstream output(files.config);
  output << "defaults:\n"
         << "  switch_back: true\n"
         << "  switch_back_mode: walkrun\n"
         << "  stop_loco_before_start: true\n"
         << "actions:\n"
         << "  " << action << ":\n"
         << "    type: fake\n"
         << "    switch_back: " << (switch_back ? "true" : "false") << "\n"
         << "    prestart_before_userctrl: true\n"
         << "    prestart_ready_timeout_sec: " << ready_timeout << "\n"
         << "    ready_file: " << files.ready.string() << "\n"
         << "    start_file: " << files.start.string() << "\n"
         << "    postrun_handoff_hold: " << (postrun_hold ? "true" : "false") << "\n"
         << "    switch_back_overlap_sec: " << switch_back_overlap_sec << "\n";
  if (postrun_hold) {
    output << "    handoff_complete_file: " << files.complete.string() << "\n"
           << "    handoff_release_file: " << files.release.string() << "\n";
  }
  output << "    command: \"" << ACTION_GO_FAKE_PAYLOAD
         << " --ready-file {ready_file} --start-file {start_file} ";
  if (postrun_hold) {
    output << "--handoff-complete-file {handoff_complete_file} "
           << "--handoff-release-file {handoff_release_file} ";
  }
  output
         << payload_args << "\"\n";
}

MotionActionController::Status wait_terminal(MotionActionController& controller,
                                             double timeout_sec = 3.0) {
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_sec));
  while (std::chrono::steady_clock::now() < deadline) {
    const auto status = controller.update();
    if (status == MotionActionController::Status::Done ||
        status == MotionActionController::Status::Failed ||
        status == MotionActionController::Status::Idle) {
      return status;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return controller.update();
}

bool wait_backend_calls(const std::atomic<int>& calls, int expected, double timeout_sec = 2.0) {
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_sec));
  while (std::chrono::steady_clock::now() < deadline) {
    if (calls.load() >= expected) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return calls.load() >= expected;
}

bool wait_status(MotionActionController& controller, MotionActionController::Status expected,
                 double timeout_sec = 2.0) {
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_sec));
  while (std::chrono::steady_clock::now() < deadline) {
    if (controller.update() == expected) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return controller.update() == expected;
}

std::shared_ptr<FakeSupervisorPlatform> fake_platform_for(const TestFiles& files) {
  auto platform = std::make_shared<FakeSupervisorPlatform>();
  platform->ready_file = files.ready.string();
  platform->start_file = files.start.string();
  platform->complete_file = files.complete.string();
  platform->release_file = files.release.string();
  return platform;
}

bool wait_fake_payload_running(const std::shared_ptr<FakeSupervisorPlatform>& platform,
                               double timeout_sec = 2.0) {
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_sec));
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(platform->mutex);
      if (platform->running && platform->start_touched) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

void test_successful_handoff() {
  TestFiles files("success");
  write_config(files, "success",
               "--environment-file " + files.environment.string() + " --run-ms 10");
  auto backend = std::make_shared<FakeControlModeBackend>();
  setenv("ACTION_GO_RUN_ID", "controller_test", 1);
  setenv("ACTION_GO_TRIAL_ID", "success", 1);
  setenv("ACTION_GO_PAIR_ID", "pair_test", 1);
  setenv("ACTION_GO_POLICY", "auto", 1);
  setenv("ACTION_GO_HANDOFF_METHOD", "auto", 1);
  setenv("ACTION_GO_GIT_SHA", "git_test", 1);
  setenv("ACTION_GO_SPEC_SHA256", "spec_test", 1);
  setenv("ACTION_GO_MODEL_SHA256", "model_test", 1);
  setenv("ACTION_GO_HARDWARE_ID", "hardware_test", 1);
  setenv("ACTION_GO_MANIFEST_SHA256", "manifest_test", 1);
  MotionActionController controller;
  controller.setEventLogPath(files.log.string());
  expect(controller.init(backend, files.config.string()), "success init");
  expect(controller.startAction("success"), "success start");
  expect(wait_terminal(controller) == MotionActionController::Status::Done, "success terminal status");
  expect(backend->stop_calls == 1, "success StopMove once");
  expect(backend->user_calls == 1, "success UserCtrl once");
  expect(backend->internal_calls == 1, "success switch-back once");
  std::ifstream log_stream(files.log);
  const std::string log((std::istreambuf_iterator<char>(log_stream)),
                        std::istreambuf_iterator<char>());
  expect(log.find("\"schema_version\":1") != std::string::npos, "event schema version");
  expect(log.find("controller_mode_confirmed") == std::string::npos,
         "RPC return is not logged as controller confirmation");
  std::ifstream environment_stream(files.environment);
  const std::string environment((std::istreambuf_iterator<char>(environment_stream)),
                                std::istreambuf_iterator<char>());
  expect(environment.find("ACTION_GO_RUN_ID=controller_test\n") != std::string::npos,
         "payload run id");
  expect(environment.find("ACTION_GO_TRIAL_ID=success_1\n") != std::string::npos,
         "payload derived trial id");
  expect(environment.find("ACTION_GO_PAIR_ID=pair_test\n") != std::string::npos,
         "payload pair id");
  expect(environment.find("ACTION_GO_POLICY=success\n") != std::string::npos,
         "payload derived policy");
  expect(environment.find("ACTION_GO_HANDOFF_METHOD=hot_standby\n") != std::string::npos,
         "payload derived handoff method");
  expect(environment.find("ACTION_GO_GIT_SHA=git_test\n") != std::string::npos,
         "payload git sha");
  expect(environment.find("ACTION_GO_SPEC_SHA256=spec_test\n") != std::string::npos,
         "payload spec hash");
  expect(environment.find("ACTION_GO_MODEL_SHA256=model_test\n") != std::string::npos,
         "payload model hash");
  expect(environment.find("ACTION_GO_HARDWARE_ID=hardware_test\n") != std::string::npos,
         "payload hardware id");
  expect(environment.find("ACTION_GO_MANIFEST_SHA256=manifest_test\n") != std::string::npos,
         "payload manifest hash");
}

void test_ready_timeout() {
  TestFiles files("timeout");
  write_config(files, "timeout", "--no-ready --run-ms 500", 0.03);
  auto backend = std::make_shared<FakeControlModeBackend>();
  MotionActionController controller;
  controller.setEventLogPath(files.log.string());
  expect(controller.init(backend, files.config.string()), "timeout init");
  expect(controller.startAction("timeout"), "timeout start");
  expect(wait_terminal(controller) == MotionActionController::Status::Failed, "ready timeout fails");
  expect(backend->user_calls == 0, "timeout does not enter UserCtrl");
  expect(backend->internal_calls == 1, "timeout restores official control");
}

void test_payload_exits_before_ready() {
  TestFiles files("early_exit");
  write_config(files, "early_exit", "--exit-before-ready --exit-code 19");
  auto backend = std::make_shared<FakeControlModeBackend>();
  MotionActionController controller;
  expect(controller.init(backend, files.config.string()), "early-exit init");
  expect(controller.startAction("early_exit"), "early-exit start");
  expect(wait_terminal(controller) == MotionActionController::Status::Failed, "early exit fails");
  expect(backend->user_calls == 0, "early exit does not enter UserCtrl");
  expect(backend->internal_calls == 1, "early exit restores official control");
}

void test_userctrl_failure_cleans_payload() {
  TestFiles files("user_fail");
  write_config(files, "user_fail",
               "--descendant-marker " + files.descendant_marker.string() +
               " --descendant-delay-ms 250 --run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  backend->fail_user = true;
  MotionActionController controller;
  expect(controller.init(backend, files.config.string()), "user-fail init");
  expect(controller.startAction("user_fail"), "user-fail start");
  expect(wait_terminal(controller) == MotionActionController::Status::Failed, "UserCtrl failure status");
  expect(backend->user_calls == 1, "UserCtrl attempted once");
  expect(backend->internal_calls == 1, "UserCtrl failure restores official control");
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  expect(!fs::exists(files.descendant_marker), "UserCtrl failure kills payload process group");
}

void test_normal_stop_and_damping() {
  {
    TestFiles files("normal_stop");
    write_config(files, "normal_stop", "--run-ms 5000");
    auto backend = std::make_shared<FakeControlModeBackend>();
    MotionActionController controller;
    controller.setEventLogPath(files.log.string());
    expect(controller.init(backend, files.config.string()), "normal-stop init");
    expect(controller.controlState() == "UNKNOWN", "unconfirmed initial control state");
    expect(controller.startAction("normal_stop"), "normal-stop start");
    expect(wait_backend_calls(backend->user_calls, 1), "normal-stop reached UserCtrl");
    expect(controller.stopAndSwitchToOfficial("walkrun"), "normal stop switches back");
    expect(controller.controlState() == "WALKRUN", "successful switch confirms WALKRUN");
    expect(backend->internal_calls == 1, "normal stop switch-back once");
    std::ifstream log_stream(files.log);
    const std::string log((std::istreambuf_iterator<char>(log_stream)),
                          std::istreambuf_iterator<char>());
    expect(log.find("external_normal_stop") != std::string::npos,
           "normal stop logs switch-back lifecycle");
    expect(log.find("action process is no longer available") == std::string::npos,
           "normal stop is recorded as cancellation, not payload failure");
  }
  {
    TestFiles files("damping");
    write_config(files, "damping", "--run-ms 5000");
    auto backend = std::make_shared<FakeControlModeBackend>();
    MotionActionController controller;
    expect(controller.init(backend, files.config.string()), "damping init");
    expect(controller.startAction("damping"), "damping start");
    expect(wait_backend_calls(backend->user_calls, 1), "damping reached UserCtrl");
    controller.dampingStop();
    expect(backend->damping_calls == 1, "damping backend called once");
    expect(controller.controlState() == "DAMPING", "successful damping confirms DAMPING");
    expect(backend->internal_calls == 0, "damping is not raced by walkrun switch-back");
  }
}

void test_running_stop_preserves_payload_through_ack_and_overlap() {
  TestFiles files("ordered_stop");
  write_config(files, "ordered_stop", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  backend->internal_delay_ms = 100;
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  platform->block_long_sleep = true;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "ordered-stop init");
  expect(controller.startAction("ordered_stop"), "ordered-stop start");
  expect(wait_fake_payload_running(platform), "ordered-stop payload running");

  std::atomic<bool> result{false};
  std::thread stop_thread([&]() { result.store(controller.stopAndSwitchToOfficial("walkrun")); });
  expect(wait_backend_calls(backend->internal_calls, 1), "ordered-stop entered 7111 RPC");
  {
    std::lock_guard<std::mutex> lock(platform->mutex);
    expect(platform->terminate_calls == 0 && platform->running,
           "payload remains active before 7111 returns");
  }
  const auto overlap_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!platform->long_sleep_entered.load() && std::chrono::steady_clock::now() < overlap_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  expect(platform->long_sleep_entered.load(), "ordered-stop entered overlap");
  expect(!controller.startAction("ordered_stop"), "normal-stop overlap rejects a new action");
  {
    std::lock_guard<std::mutex> lock(platform->mutex);
    expect(platform->terminate_calls == 0 && platform->running,
           "payload remains active throughout overlap");
  }
  platform->allow_long_sleep.store(true);
  stop_thread.join();
  expect(result.load(), "ordered-stop succeeds");
  expect(platform->terminate_calls == 1, "payload terminates after overlap");
}

void test_configured_overlap_window_is_preserved() {
  TestFiles files("configured_overlap");
  constexpr double overlap_sec = 1.0;
  write_config(files, "configured_overlap", "--run-ms 5000", 0.5, true, false,
               overlap_sec);
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  platform->block_long_sleep = true;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()),
         "configured-overlap init");
  expect(controller.startAction("configured_overlap"),
         "configured-overlap start");
  expect(wait_fake_payload_running(platform), "configured-overlap payload running");

  std::atomic<bool> stop_result{false};
  std::thread stop_thread([&]() {
    stop_result.store(controller.stopAndSwitchToOfficial("walkrun"));
  });
  expect(wait_backend_calls(backend->internal_calls, 1),
         "configured-overlap entered 7111 RPC");
  const auto overlap_deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(2);
  while (!platform->long_sleep_entered.load() &&
         std::chrono::steady_clock::now() < overlap_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  expect(platform->long_sleep_entered.load(),
         "configured-overlap entered configured overlap");
  expect(platform->long_sleep_duration_ms.load() >= 1000,
         "configured-overlap uses one-second window");
  platform->allow_long_sleep.store(true);
  stop_thread.join();
  expect(stop_result.load(), "configured-overlap stop succeeds");
  {
    std::lock_guard<std::mutex> lock(platform->mutex);
    expect(platform->terminate_time == platform->current_time,
           "configured-overlap terminates after overlap");
    expect(platform->terminate_calls == 1,
           "configured-overlap terminates payload once");
  }
}

void test_postrun_hold_release_and_retry() {
  {
    TestFiles files("postrun_hold");
    write_config(files, "postrun_hold", "--run-ms 10", 0.5, true, true);
    auto backend = std::make_shared<FakeControlModeBackend>();
    auto platform = fake_platform_for(files);
    platform->complete_after_start = false;
    platform->hold_after_start = true;
    MotionActionController controller;
    expect(controller.init(backend, platform, files.config.string()), "postrun-hold init");
    expect(controller.startAction("postrun_hold"), "postrun-hold start");
    expect(wait_terminal(controller) == MotionActionController::Status::Done,
           "postrun hold completes after release");
    expect(platform->files.count(files.release.string()) == 1, "postrun release touched");
    expect(platform->terminate_calls == 0, "normal completed hold exits without SIGTERM");
    expect(backend->internal_calls == 1, "completed hold switches once");
  }
  {
    TestFiles files("postrun_retry");
    write_config(files, "postrun_retry", "--run-ms 10", 0.5, true, true);
    auto backend = std::make_shared<FakeControlModeBackend>();
    backend->fail_internal = true;
    auto platform = fake_platform_for(files);
    platform->complete_after_start = false;
    platform->hold_after_start = true;
    MotionActionController controller;
    expect(controller.init(backend, platform, files.config.string()), "postrun-retry init");
    expect(controller.startAction("postrun_retry"), "postrun-retry start");
    expect(wait_status(controller, MotionActionController::Status::SwitchBackBlocked),
           "failed 7111 enters retryable blocked state");
    {
      std::lock_guard<std::mutex> lock(platform->mutex);
      expect(platform->running, "blocked payload keeps running");
      expect(platform->files.count(files.release.string()) == 0, "blocked hold is not released");
      expect(platform->terminate_calls == 0, "blocked payload is not terminated");
    }
    backend->fail_internal = false;
    expect(controller.stopAndSwitchToOfficial("walkrun"), "blocked handoff retry succeeds");
    expect(controller.update() == MotionActionController::Status::Idle, "retry returns idle");
    expect(backend->internal_calls == 2, "retry makes second 7111 call");
    expect(platform->terminate_calls == 0, "retry releases hold without SIGTERM");
  }
}

void test_running_switch_failure_keeps_payload_for_retry() {
  TestFiles files("running_retry");
  write_config(files, "running_retry", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  backend->fail_internal = true;
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "running-retry init");
  expect(controller.startAction("running_retry"), "running-retry start");
  expect(wait_fake_payload_running(platform), "running-retry payload running");
  expect(!controller.stopAndSwitchToOfficial("walkrun"), "running-retry first 7111 fails");
  expect(controller.update() == MotionActionController::Status::SwitchBackBlocked,
         "running-retry enters blocked state");
  {
    std::lock_guard<std::mutex> lock(platform->mutex);
    expect(platform->running && platform->terminate_calls == 0,
           "running payload survives failed 7111");
  }
  backend->fail_internal = false;
  expect(controller.stopAndSwitchToOfficial("walkrun"), "running-retry second 7111 succeeds");
  expect(platform->terminate_calls == 1, "running payload stops only after successful overlap");
}

void test_blocked_handoff_damping_preempts() {
  TestFiles files("blocked_damping");
  write_config(files, "blocked_damping", "--run-ms 10", 0.5, true, true);
  auto backend = std::make_shared<FakeControlModeBackend>();
  backend->fail_internal = true;
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  platform->hold_after_start = true;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "blocked-damping init");
  expect(controller.startAction("blocked_damping"), "blocked-damping start");
  expect(wait_status(controller, MotionActionController::Status::SwitchBackBlocked),
         "blocked-damping reaches blocked state");
  controller.dampingStop();
  expect(platform->terminate_calls == 1, "damping immediately terminates blocked hold");
  expect(backend->damping_calls == 1, "damping owns controller after blocked hold");
}

void test_handoff_overlap_damping_preempts() {
  TestFiles files("overlap_damping");
  write_config(files, "overlap_damping", "--run-ms 10", 0.5, true, true);
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  platform->hold_after_start = true;
  platform->block_long_sleep = true;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "overlap-damping init");
  expect(controller.startAction("overlap_damping"), "overlap-damping start");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!platform->long_sleep_entered.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  expect(platform->long_sleep_entered.load(), "overlap-damping reaches handoff overlap");
  std::thread damping_thread([&]() { controller.dampingStop(); });
  expect(wait_backend_calls(backend->damping_calls, 1), "overlap damping enters damping immediately");
  {
    std::lock_guard<std::mutex> lock(platform->mutex);
    expect(platform->terminate_calls == 1 && platform->last_terminate_forced,
           "overlap damping force-terminates hold");
  }
  platform->allow_long_sleep.store(true);
  damping_thread.join();
  expect(controller.update() == MotionActionController::Status::Failed,
         "overlap damping remains stopped");
}

void test_normal_stop_before_ready() {
  TestFiles files("normal_stop_before_ready");
  write_config(files, "normal_stop_before_ready", "--no-ready --run-ms 5000", 2.0);
  auto backend = std::make_shared<FakeControlModeBackend>();
  MotionActionController controller;
  expect(controller.init(backend, files.config.string()), "pre-ready normal-stop init");
  expect(controller.startAction("normal_stop_before_ready"), "pre-ready normal-stop start");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  expect(controller.stopAndSwitchToOfficial("walkrun"), "pre-ready normal stop switches back");
  expect(backend->user_calls == 0, "pre-ready normal stop never enters UserCtrl");
  expect(backend->internal_calls == 1, "pre-ready normal stop switches to walkrun once");
  expect(controller.update() == MotionActionController::Status::Idle,
         "pre-ready normal stop returns controller to idle");
}

void test_normal_stop_during_userctrl_handoff() {
  TestFiles files("normal_stop_during_userctrl");
  write_config(files, "normal_stop_during_userctrl", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  backend->user_delay_ms = 100;
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()),
         "handoff-race normal-stop init");
  expect(controller.startAction("normal_stop_during_userctrl"),
         "handoff-race normal-stop start");
  expect(wait_backend_calls(backend->user_calls, 1), "handoff-race entered UserCtrl RPC");
  expect(controller.stopAndSwitchToOfficial("walkrun"),
         "handoff-race normal stop switches back");
  expect(!backend->active_user_ctrl.load(), "handoff-race final mode is official");
  expect(backend->internal_calls == 1, "handoff-race switches to walkrun once");
  expect(controller.update() == MotionActionController::Status::Idle,
         "handoff-race normal stop returns controller to idle");
}

void test_injected_start_file_failure_falls_back() {
  TestFiles files("start_file_failure");
  write_config(files, "start_file_failure", "--run-ms 10", 0.5, false);
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->fail_touch = true;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "start-file-fail init");
  expect(controller.startAction("start_file_failure"), "start-file-fail start");
  expect(wait_terminal(controller) == MotionActionController::Status::Failed,
         "start-file failure status");
  expect(backend->user_calls == 1, "start-file failure entered UserCtrl");
  expect(backend->internal_calls == 1, "start-file failure restored official control");
  expect(platform->terminate_calls == 1, "start-file failure cleaned process group");
}

void test_injected_stale_barriers_are_removed() {
  TestFiles files("stale_barriers");
  write_config(files, "stale_barriers", "--run-ms 10", 0.03, true, true);
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->provide_ready = false;
  platform->files.insert(files.ready.string());
  platform->files.insert(files.start.string());
  platform->files.insert(files.complete.string());
  platform->files.insert(files.release.string());
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "stale-barrier init");
  expect(controller.startAction("stale_barriers"), "stale-barrier start");
  expect(wait_terminal(controller) == MotionActionController::Status::Failed,
         "stale ready file cannot satisfy new trial");
  expect(platform->removed_files.count(files.ready.string()) == 1,
         "stale ready file removed");
  expect(platform->removed_files.count(files.start.string()) == 1,
         "stale start file removed");
  expect(platform->removed_files.count(files.complete.string()) == 1,
         "stale complete file removed");
  expect(platform->removed_files.count(files.release.string()) == 1,
         "stale release file removed");
  expect(backend->user_calls == 0, "stale barriers do not enter UserCtrl");
  expect(platform->terminate_calls == 1, "ready timeout cleans process group");
}

void test_injected_runtime_failure_always_falls_back() {
  TestFiles files("runtime_failure");
  write_config(files, "runtime_failure", "--run-ms 10", 0.5, false);
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->completion_exit_code = 23;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "runtime-failure init");
  expect(controller.startAction("runtime_failure"), "runtime-failure start");
  expect(wait_terminal(controller) == MotionActionController::Status::Failed,
         "runtime-failure status");
  expect(backend->user_calls == 1, "runtime failure entered UserCtrl");
  expect(backend->internal_calls == 1,
         "runtime failure restored official control despite switch_back false");
}

void test_injected_repeated_switching() {
  TestFiles files("repeated");
  write_config(files, "repeated", "--run-ms 10");
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "repeated init");
  for (int trial = 0; trial < 3; ++trial) {
    expect(controller.startAction("repeated"), "repeated start");
    expect(wait_terminal(controller) == MotionActionController::Status::Done,
           "repeated terminal status");
  }
  expect(backend->user_calls == 3, "repeated UserCtrl count");
  expect(backend->internal_calls == 3, "repeated switch-back count");
  expect(platform->spawn_calls == 3, "repeated payload spawn count");
}

void test_injected_cancel_cleans_process() {
  TestFiles files("cancel");
  write_config(files, "cancel", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "cancel init");
  expect(controller.startAction("cancel"), "cancel start");
  expect(wait_backend_calls(backend->user_calls, 1), "cancel reached UserCtrl");
  controller.cancel();
  expect(platform->terminate_calls == 1, "cancel cleaned process group");
  expect(backend->internal_calls == 1, "cancel restored official control");
  expect(controller.update() == MotionActionController::Status::Idle, "cancel is idle");
}

void test_injected_dry_cancel_is_not_fault() {
  TestFiles files("dry_cancel");
  write_config(files, "dry_cancel", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string(), "", true),
         "dry-cancel init");
  expect(controller.startAction("dry_cancel"), "dry-cancel start");
  expect(wait_fake_payload_running(platform), "dry-cancel payload running");
  controller.cancel();
  expect(controller.update() == MotionActionController::Status::Idle,
         "dry cancel is idle");
  expect(!controller.hasError(), "dry cancel is not a fault");
}

void test_injected_dry_postrun_hold_releases_without_rpc() {
  TestFiles files("dry_postrun_hold");
  write_config(files, "dry_postrun_hold", "--run-ms 10", 0.5, true, true);
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  platform->hold_after_start = true;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string(), "", true),
         "dry-postrun init");
  expect(controller.startAction("dry_postrun_hold"), "dry-postrun start");
  expect(wait_terminal(controller) == MotionActionController::Status::Done,
         "dry-postrun releases completed hold");
  expect(backend->user_calls == 0 && backend->internal_calls == 0,
         "dry-postrun makes no control RPC");
  expect(platform->files.count(files.release.string()) == 1,
         "dry-postrun release file touched");
}

void test_damping_stop_allows_restart() {
  TestFiles files("damping_restart");
  write_config(files, "damping_restart", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "damping-restart init");
  expect(controller.startAction("damping_restart"), "damping-restart start");
  expect(wait_backend_calls(backend->user_calls, 1), "damping-restart reached UserCtrl");
  controller.dampingStop();
  expect(backend->damping_calls == 1, "stop enters damping");
  expect(controller.update() == MotionActionController::Status::Failed, "damping stops action");
  expect(controller.startAction("damping_restart"), "damping allows explicit restart");
  expect(wait_backend_calls(backend->user_calls, 2), "restart reaches UserCtrl");
  controller.dampingStop();
}

void test_motion_config_rejects_ambiguous_input() {
  struct Case {
    const char* name;
    const char* body;
  };
  const Case cases[] = {
      {"invalid_bool", "actions:\n  bad:\n    type: fake\n    safety_enable: flase\n"},
      {"unknown_field", "actions:\n  bad:\n    type: fake\n    commnad: echo bad\n"},
      {"invalid_mode", "actions:\n  bad:\n    type: fake\n    switch_back_mode: pasive\n"},
      {"missing_colon", "actions:\n  bad\n    type: fake\n"},
      {"duplicate_id", "actions:\n  first:\n    type: fake\n  first:\n    type: fake\n"},
      {"unknown_section", "unexpected:\n  value: true\n"},
  };

  for (const auto& test_case : cases) {
    TestFiles files(std::string("config_") + test_case.name);
    {
      std::ofstream output(files.config);
      output << test_case.body;
    }
    auto backend = std::make_shared<FakeControlModeBackend>();
    MotionActionController controller;
    bool rejected = false;
    try {
      (void)controller.init(backend, files.config.string());
    } catch (const std::exception&) {
      rejected = true;
    }
    expect(rejected, std::string("config rejects ") + test_case.name);
  }
}

void test_concurrent_start_is_serialized() {
  TestFiles files("concurrent_start");
  write_config(files, "concurrent", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "concurrent init");
  std::atomic<int> waiting{0};
  std::atomic<bool> go{false};
  std::atomic<int> accepted{0};
  auto start = [&]() {
    ++waiting;
    while (!go.load()) std::this_thread::yield();
    if (controller.startAction("concurrent")) ++accepted;
  };
  std::thread first(start);
  std::thread second(start);
  while (waiting.load() != 2) std::this_thread::yield();
  go.store(true);
  first.join();
  second.join();
  expect(accepted == 1, "only one concurrent start accepted");
  expect(wait_status(controller, MotionActionController::Status::Running),
         "concurrent payload reaches running");
  controller.cancel();
  expect(platform->spawn_calls == 1, "only one payload spawned");
}

void test_private_default_handshake() {
  TestFiles files("private_handshake");
  {
    std::ofstream output(files.config);
    output << "actions:\n  private:\n    type: fake\n    prestart_before_userctrl: true\n"
           << "    command: \"" << ACTION_GO_FAKE_PAYLOAD
           << " --ready-file {ready_file} --start-file {start_file} --run-ms 5000\"\n";
  }
  auto backend = std::make_shared<FakeControlModeBackend>();
  MotionActionController controller;
  controller.setEventLogPath(files.log.string());
  expect(controller.init(backend, files.config.string()), "private handshake init");
  std::string previous_directory;
  for (int attempt = 1; attempt <= 2; ++attempt) {
    expect(controller.startAction("private"), "private handshake start");
    expect(wait_backend_calls(backend->user_calls, attempt), "private handshake ready");
    controller.flushLogs();
    std::ifstream input(files.log);
    std::string line;
    std::string ready;
    while (std::getline(input, line)) {
      const auto event = nlohmann::json::parse(line);
      if (event.value("event", "") == "action_triggered") ready = event.value("ready_file", "");
    }
    const auto directory = fs::path(ready).parent_path().string();
    struct stat info {};
    expect(!ready.empty() && stat(directory.c_str(), &info) == 0 &&
           (info.st_mode & 0777) == 0700, "private handshake directory is 0700");
    expect(directory != previous_directory, "fresh directory for each activation");
    controller.cancel();
    expect(!fs::exists(directory), "private directory cleaned after child exits");
    previous_directory = directory;
  }
}

void test_signal_files_reject_links() {
  TestFiles files("signals");
  const auto target = files.root / "target";
  { std::ofstream output(target); output << "preserve"; }
  fs::create_symlink(target, files.ready);
  auto platform = action_go::runtime::make_posix_supervisor_platform();
  std::string error;
  expect(!platform->touch_file(files.ready.string(), "bad", error), "reject signal symlink write");
  expect(!platform->file_exists(files.ready.string()), "reject signal symlink read");
  std::ifstream input(target);
  std::string contents;
  input >> contents;
  expect(contents == "preserve", "symlink target unchanged");
  fs::remove(files.ready);
  expect(platform->touch_file(files.ready.string(), "ready", error), "create signal");
  expect(platform->touch_file(files.ready.string(), "ready", error), "repeated signal is idempotent");
  action_go::runtime::write_signal_file(files.ready.string(), "first reason\n", true);
  action_go::runtime::write_signal_file(files.ready.string(), "new\n", true);
  {
    std::ifstream updated(files.ready);
    std::string reason((std::istreambuf_iterator<char>(updated)), std::istreambuf_iterator<char>());
    expect(reason == "new\n", "atomic signal update replaces old reason without trailing content");
  }
  fs::create_hard_link(target, files.start);
  expect(!platform->touch_file(files.start.string(), "bad", error), "reject hard-linked signal");
  for (const auto& linked : {files.start, files.root / "symlink"}) {
    if (!fs::exists(linked)) fs::create_symlink(target, linked);
    bool rejected = false;
    try {
      action_go::runtime::write_signal_file(linked.string(), "bad", true);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    expect(rejected, "atomic signal update rejects linked destination");
  }
  {
    std::ifstream preserved(target);
    std::string value;
    preserved >> value;
    expect(value == "preserve", "atomic update leaves linked target unchanged");
  }
  const auto link = files.root / "directory_link";
  fs::create_directory_symlink(files.root, link);
  expect(!platform->touch_file((link / "new_signal").string(), "bad", error), "reject linked parent");
}

void test_command_hash_and_empty_placeholder() {
  TestFiles files("command_validation");
  {
    std::ofstream output(files.config);
    output << "actions:\n  hash:\n    type: fake\n    command: \"printf '%s' '#kept' > "
           << files.environment.string() << " {event_log_arg}\" # comment\n"
           << "  empty:\n    type: fake\n    command: \"echo --handoff-complete-file {handoff_complete_file} --dry-run\"\n";
  }
  auto backend = std::make_shared<FakeControlModeBackend>();
  MotionActionController controller;
  expect(controller.init(backend, files.config.string()), "command validation init");
  expect(!controller.startAction("empty"), "reject empty required placeholder");
  expect(backend->user_calls == 0, "empty placeholder rejected before UserCtrl");
  expect(controller.startAction("hash"), "hash command starts");
  expect(wait_terminal(controller) == MotionActionController::Status::Done, "hash command completes");
  std::ifstream input(files.environment);
  std::string contents;
  input >> contents;
  expect(contents == "#kept", "quoted hash preserved and optional placeholder accepted");
}

}  // namespace

int main() {
  test_successful_handoff();
  test_ready_timeout();
  test_payload_exits_before_ready();
  test_userctrl_failure_cleans_payload();
  test_normal_stop_and_damping();
  test_running_stop_preserves_payload_through_ack_and_overlap();
  test_configured_overlap_window_is_preserved();
  test_postrun_hold_release_and_retry();
  test_running_switch_failure_keeps_payload_for_retry();
  test_blocked_handoff_damping_preempts();
  test_handoff_overlap_damping_preempts();
  test_normal_stop_before_ready();
  test_normal_stop_during_userctrl_handoff();
  test_injected_start_file_failure_falls_back();
  test_injected_stale_barriers_are_removed();
  test_injected_runtime_failure_always_falls_back();
  test_injected_repeated_switching();
  test_injected_cancel_cleans_process();
  test_injected_dry_cancel_is_not_fault();
  test_injected_dry_postrun_hold_releases_without_rpc();
  test_damping_stop_allows_restart();
  test_motion_config_rejects_ambiguous_input();
  test_concurrent_start_is_serialized();
  test_private_default_handshake();
  test_signal_files_reject_links();
  test_command_hash_and_empty_placeholder();
  if (failures != 0) {
    std::cerr << "[motion_action_controller_test] failures=" << failures << '\n';
    return 1;
  }
  std::cout << "[motion_action_controller_test] OK\n";
  return 0;
}
