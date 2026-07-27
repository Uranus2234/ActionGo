#include "action_go/motion_action_controller.h"
#include "action_go/runtime/control_mode_backend.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    return {};
  }

  bool terminate_process_group(pid_t, bool force,
                               std::string&) override {
    std::lock_guard<std::mutex> lock(mutex);
    ++terminate_calls;
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
  std::string last_command;
  bool provide_ready{true};
  bool fail_remove{false};
  bool fail_touch{false};
  bool fail_spawn{false};
  bool exit_before_ready{false};
  bool complete_after_start{true};
  bool start_touched{false};
  bool running{false};
  bool last_terminate_forced{false};
  int remove_calls{0};
  int touch_calls{0};
  int spawn_calls{0};
  int poll_calls{0};
  int terminate_calls{0};
  int completion_exit_code{0};
  pid_t next_pid{10000};
};

struct TestFiles {
  fs::path root;
  fs::path config;
  fs::path log;
  fs::path environment;
  fs::path ready;
  fs::path start;
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
    descendant_marker = root / "descendant_survived";
  }

  ~TestFiles() {
    std::error_code error;
    fs::remove_all(root, error);
  }
};

void write_config(const TestFiles& files, const std::string& action,
                  const std::string& payload_args, double ready_timeout = 0.5,
                  bool switch_back = true) {
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
         << "    command: \"" << ACTION_GO_FAKE_PAYLOAD
         << " --ready-file {ready_file} --start-file {start_file} "
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

std::shared_ptr<FakeSupervisorPlatform> fake_platform_for(const TestFiles& files) {
  auto platform = std::make_shared<FakeSupervisorPlatform>();
  platform->ready_file = files.ready.string();
  platform->start_file = files.start.string();
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
  expect(log.find("controller_mode_confirmed") != std::string::npos, "mode confirmation event");
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
    expect(controller.startAction("normal_stop"), "normal-stop start");
    expect(wait_backend_calls(backend->user_calls, 1), "normal-stop reached UserCtrl");
    expect(controller.stopAndSwitchToOfficial("walkrun"), "normal stop switches back");
    expect(backend->internal_calls == 1, "normal stop switch-back once");
    std::ifstream log_stream(files.log);
    const std::string log((std::istreambuf_iterator<char>(log_stream)),
                          std::istreambuf_iterator<char>());
    expect(log.find("external_normal_stop") != std::string::npos,
           "normal stop logs switch-back lifecycle");
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
    expect(backend->internal_calls == 0, "damping is not raced by walkrun switch-back");
  }
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
  write_config(files, "stale_barriers", "--run-ms 10", 0.03);
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->provide_ready = false;
  platform->files.insert(files.ready.string());
  platform->files.insert(files.start.string());
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "stale-barrier init");
  expect(controller.startAction("stale_barriers"), "stale-barrier start");
  expect(wait_terminal(controller) == MotionActionController::Status::Failed,
         "stale ready file cannot satisfy new trial");
  expect(platform->removed_files.count(files.ready.string()) == 1,
         "stale ready file removed");
  expect(platform->removed_files.count(files.start.string()) == 1,
         "stale start file removed");
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

void test_emergency_stop_latches_until_local_reset() {
  TestFiles files("emergency_latch");
  write_config(files, "emergency_latch", "--run-ms 5000");
  auto backend = std::make_shared<FakeControlModeBackend>();
  auto platform = fake_platform_for(files);
  platform->complete_after_start = false;
  MotionActionController controller;
  expect(controller.init(backend, platform, files.config.string()), "emergency-latch init");
  expect(controller.startAction("emergency_latch"), "emergency-latch start");
  expect(wait_backend_calls(backend->user_calls, 1), "emergency-latch reached UserCtrl");
  controller.emergencyStop();
  expect(controller.emergencyLatched(), "emergency latch set");
  expect(backend->damping_calls == 1, "emergency enters damping");
  expect(!controller.startAction("emergency_latch"), "latched emergency rejects action");
  controller.clearEmergencyLatch();
  expect(!controller.emergencyLatched(), "local reset clears emergency latch");
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

}  // namespace

int main() {
  test_successful_handoff();
  test_ready_timeout();
  test_payload_exits_before_ready();
  test_userctrl_failure_cleans_payload();
  test_normal_stop_and_damping();
  test_normal_stop_before_ready();
  test_normal_stop_during_userctrl_handoff();
  test_injected_start_file_failure_falls_back();
  test_injected_stale_barriers_are_removed();
  test_injected_runtime_failure_always_falls_back();
  test_injected_repeated_switching();
  test_injected_cancel_cleans_process();
  test_injected_dry_cancel_is_not_fault();
  test_emergency_stop_latches_until_local_reset();
  test_motion_config_rejects_ambiguous_input();
  if (failures != 0) {
    std::cerr << "[motion_action_controller_test] failures=" << failures << '\n';
    return 1;
  }
  std::cout << "[motion_action_controller_test] OK\n";
  return 0;
}
