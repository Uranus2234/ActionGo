#ifndef MOTION_ACTION_CONTROLLER_H
#define MOTION_ACTION_CONTROLLER_H

#include "action_go/runtime/control_mode_backend.h"
#include "action_go/runtime/event_logger.h"
#include "action_go/runtime/supervisor_platform.h"

#include <atomic>
#include <csignal>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace unitree::robot::g1 {
class LocoClient;
}

class MotionActionController
{
public:
    enum class Status
    {
        Idle,
        SwitchingToUserCtrl,
        Running,
        SwitchingBack,
        SwitchBackBlocked,
        Done,
        Failed
    };

    struct ActionConfig
    {
        std::string name;
        std::string type{"beyondmimic"};
        std::string model;
        std::string params;
        std::string trajectory;
        std::string command;
        // Directory of the YAML file where this action was declared.  Used so
        // resource paths in split action files can be relative to that file.
        std::string resource_base_dir;
        // Optional command run before switching to UserCtrl.  This is intended
        // for DDS/camera/ONNX dry-runs that do not publish LowCmd.  If it
        // returns non-zero, the real payload is not started.
        std::string preflight_command;
        std::string network_interface;
        double duration_sec{0.0};
        // Optional policy/motion frame limit for players that support it.
        // 0 means no explicit limit.
        int max_frames{0};
        bool switch_back{true};
        bool stop_loco_before_start{true};
        // Per-action IMU safety gate used by action_go_remote_test.  The
        // global remote safety switch still has priority; this only lets a
        // specific motion opt out of being interrupted by the safety recovery.
        bool safety_enable{true};
        // Start the external payload before switching to UserCtrl.  The payload
        // should create ready_file when it is publishing a safe hold command and
        // wait for start_file before starting the real policy.  This removes the
        // SwitchToUserCtrl -> first LowCmd gap for heavier players.
        bool prestart_before_userctrl{false};
        double prestart_ready_timeout_sec{3.0};
        std::string ready_file;
        std::string start_file;
        bool postrun_handoff_hold{false};
        std::string handoff_complete_file;
        std::string handoff_release_file;
        double switch_back_overlap_sec{0.30};
        std::string supervisor_event_log;
        std::string payload_event_log;
        std::string event_log_arg;
        std::string trial_id;
        uint64_t action_seq{0};
        // Which official FSM mode to enter after action:
        // "walkrun" = normal official walk/run controller,
        // "passive" = official damping,
        // "last" = previous official mode.
        std::string switch_back_mode{"walkrun"};
    };

    MotionActionController();
    ~MotionActionController();

    MotionActionController(const MotionActionController&) = delete;
    MotionActionController& operator=(const MotionActionController&) = delete;

    bool init(unitree::robot::g1::LocoClient* loco_client,
              const std::string& config_path,
              const std::string& network_interface_override = "",
              bool dry_run = false);
    bool init(std::shared_ptr<action_go::runtime::ControlModeBackend> control_backend,
              const std::string& config_path,
              const std::string& network_interface_override = "",
              bool dry_run = false);
    bool init(std::shared_ptr<action_go::runtime::ControlModeBackend> control_backend,
              std::shared_ptr<action_go::runtime::SupervisorPlatform> supervisor_platform,
              const std::string& config_path,
              const std::string& network_interface_override = "",
              bool dry_run = false);
    void setEventLogPath(const std::string& path);
    void flushLogs() { event_log_.flush(); }
    bool logsHealthy() const { return event_log_.healthy(); }
    unsigned long long droppedLogRecords() const { return event_log_.dropped(); }
    std::string currentActionEventLog() const;
    std::string controlState() const;

    bool startAction(const std::string& action_name);
    Status update();
    void cancel();
    // Safety recovery path: stop the current UserCtrl payload without first
    // switching back to official walkrun/passive, so a recovery action can be
    // started immediately.
    void interruptForRecovery();
    void dampingStop();
    bool stopAndSwitchToOfficial(const std::string& mode = "walkrun");

    bool isRunning() const;
    bool isDone() const;
    bool hasError() const;
    std::string lastError() const;
    std::string currentAction() const;
    bool hasAction(const std::string& action_name) const;
    bool actionSafetyEnabled(const std::string& action_name) const;

    static const char* statusToString(Status status);

private:
    bool loadConfig(const std::string& config_path);
    void workerMain(ActionConfig action);
    bool switchToUserCtrl(std::string& error_message, bool publish_prearm = true);
    bool switchBackToOfficial(std::string& error_message,
                              const std::string& mode);
    bool enterOfficialDamping(std::string& error_message);
    bool runActionPayload(const ActionConfig& action, std::string& error_message);
    bool runExternalCommand(const ActionConfig& action, std::string& error_message);
    bool runCommandString(const std::string& command, std::string& error_message);
    bool runCommandString(const ActionConfig& action,
                          const std::string& command,
                          std::string& error_message,
                          const std::string& phase);
    bool startCommandStringAsync(const std::string& command, std::string& error_message);
    bool waitRunningCommand(const std::string& command, std::string& error_message,
                            const ActionConfig* action = nullptr);
    bool waitForFile(const std::string& path, double timeout_sec, std::string& error_message);
    bool touchFile(const std::string& path, std::string& error_message);
    void prepareActionEventLog(ActionConfig& action);
    void logActionEvent(const std::string& event, const ActionConfig& action,
                        const std::string& reason = "");
    bool switchBackToOfficialLogged(std::string& error_message,
                                    const std::string& mode,
                                    const ActionConfig& action,
                                    const std::string& reason);
    bool completeSwitchBack(const ActionConfig& action,
                            const std::string& mode,
                            const std::string& reason,
                            std::string& error_message);
    ActionConfig currentActionConfig() const;
    void terminateRunningProcess(bool force);
    void joinWorkerIfJoinable();
    bool waitDurationOrCancel(double duration_sec);
    void setStatus(Status status);
    void setFailed(const std::string& error_message);
    std::string expandCommandTemplate(const ActionConfig& action) const;
    std::string expandCommandTemplate(const ActionConfig& action,
                                      const std::string& command_template) const;
    std::string dryRunCommandString(const ActionConfig& action) const;
    std::string resolveResourcePath(const std::string& raw_path) const;
    std::string resolveResourcePath(const std::string& raw_path,
                                    const std::string& base_dir) const;
    std::string defaultPlayerPath() const;

    std::shared_ptr<action_go::runtime::ControlModeBackend> control_backend_;
    std::shared_ptr<action_go::runtime::SupervisorPlatform> supervisor_platform_;
    std::filesystem::path config_dir_;
    std::filesystem::path action_go_root_;
    std::string network_interface_override_;
    bool dry_run_{false};
    std::string event_log_path_;
    action_go::runtime::RuntimeEventLogger event_log_;
    action_go::runtime::ActionLifecycleEventV1 base_event_context_;
    uint64_t next_action_seq_{0};
    std::unordered_map<std::string, ActionConfig> actions_;

    mutable std::mutex mutex_;
    std::mutex start_mutex_;
    bool official_stop_in_progress_{false};
    std::mutex worker_mutex_;
    std::mutex process_mutex_;
    std::mutex control_mode_mutex_;
    std::mutex switch_back_mutex_;
    std::thread worker_;
    std::atomic<bool> cancel_requested_{false};
    std::atomic<bool> suppress_switch_back_on_cancel_{false};
    std::atomic<bool> handoff_release_sent_{false};
    std::atomic<Status> status_{Status::Idle};
    std::atomic<pid_t> running_pid_{-1};
    std::atomic<int> control_state_{0};
    std::string current_action_;
    ActionConfig active_action_;
    std::string handshake_directory_;
    uint64_t current_action_seq_{0};
    std::string current_trial_id_;
    action_go::runtime::ActionLifecycleEventV1 current_event_context_;
    std::string current_switch_back_mode_{"walkrun"};
    std::string last_error_;
};

#endif // MOTION_ACTION_CONTROLLER_H
