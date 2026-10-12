#include "action_go/motion_action_controller.h"
#include "action_go/runtime/prearm_command.h"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/robot/b2/robot_state/robot_state_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <cmath>
#include <memory>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;
using action_go::runtime::event_bool;
using action_go::runtime::event_int;
using action_go::runtime::event_number;
using action_go::runtime::event_string;

namespace
{
constexpr const char* kUserLowCmdTopic = "rt/user_lowcmd";
constexpr int kG1MotorCount = 29;

std::string trim(const std::string& input)
{
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

std::string unquote(std::string value)
{
    value = trim(value);
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

bool parseBool(const std::string& value, bool default_value)
{
    std::string lower = trim(value);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "true" || lower == "1" || lower == "yes" || lower == "on") {
        return true;
    }
    if (lower == "false" || lower == "0" || lower == "no" || lower == "off") {
        return false;
    }
    throw std::runtime_error("invalid boolean value: " + value);
}

bool isNumericLiteral(const std::string& value)
{
    const std::string s = trim(value);
    if (s.empty()) {
        return false;
    }

    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && errno != ERANGE && std::isfinite(parsed);
}

void replaceAll(std::string& text, const std::string& from, const std::string& to)
{
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string shellQuote(const std::string& input)
{
    std::string out = "'";
    for (char c : input) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

std::string safeFileComponent(const std::string& input)
{
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            out += c;
        } else {
            out += '_';
        }
    }
    return out.empty() ? "action" : out;
}

template <typename T, typename = void>
struct HasSwitchToUserCtrl : std::false_type {};

template <typename T>
struct HasSwitchToUserCtrl<T, std::void_t<decltype(std::declval<T&>().SwitchToUserCtrl())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSwitchToInternalCtrl : std::false_type {};

template <typename T>
struct HasSwitchToInternalCtrl<T, std::void_t<decltype(&T::SwitchToInternalCtrl)>>
    : std::true_type {};

template <typename>
struct SwitchToInternalCtrlArg {};

template <typename R, typename C, typename Arg>
struct SwitchToInternalCtrlArg<R (C::*)(Arg)>
{
    using type = Arg;
};

template <typename ClientT>
bool callNativeSwitchToUserCtrl(ClientT* client, std::string& error_message)
{
    if constexpr (HasSwitchToUserCtrl<ClientT>::value) {
        const int32_t ret = client->SwitchToUserCtrl();
        if (ret != 0) {
            error_message = "LocoClient::SwitchToUserCtrl failed ret=" + std::to_string(ret);
            return false;
        }
        return true;
    } else {
        (void)client;
        error_message = "LocoClient::SwitchToUserCtrl is not available in this SDK header";
        return false;
    }
}

template <typename ClientT>
bool callNativeSwitchToInternal(ClientT* client, int mode_id, std::string& error_message)
{
    if constexpr (HasSwitchToInternalCtrl<ClientT>::value) {
        using Arg = typename SwitchToInternalCtrlArg<decltype(&ClientT::SwitchToInternalCtrl)>::type;
        // Unitree SDK defines InternalFsmMode as:
        // LAST=0, PASSIVE=1, WALKRUN=2.  Use static_cast
        // here so this file still compiles with SDK headers where the enum is
        // not declared in this translation unit.
        const int32_t ret = client->SwitchToInternalCtrl(static_cast<Arg>(mode_id));
        if (ret != 0) {
            error_message = "LocoClient::SwitchToInternalCtrl(" + std::to_string(mode_id) +
                            ") failed ret=" + std::to_string(ret);
            return false;
        }
        return true;
    } else {
        (void)client;
        error_message = "LocoClient::SwitchToInternalCtrl is not available in this SDK header";
        return false;
    }
}

std::string normalizeInternalMode(std::string mode)
{
    mode = trim(mode);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "walk" || mode == "run" || mode == "walk_run" || mode == "walk-run") {
        return "walkrun";
    }
    if (mode == "damp" || mode == "damping") {
        return "passive";
    }
    if (mode == "0") {
        return "last";
    }
    if (mode == "1") {
        return "passive";
    }
    if (mode == "2") {
        return "walkrun";
    }
    if (mode == "last" || mode == "passive" || mode == "walkrun") {
        return mode;
    }
    throw std::runtime_error("invalid switch_back_mode: " + mode);
}

void applyActionField(MotionActionController::ActionConfig& action,
                      const std::string& key,
                      const std::string& value)
{
    if (key == "type") {
        action.type = value;
    } else if (key == "model") {
        action.model = value;
    } else if (key == "params") {
        action.params = value;
    } else if (key == "trajectory") {
        action.trajectory = value;
    } else if (key == "command") {
        action.command = value;
    } else if (key == "preflight_command" || key == "dry_run_command") {
        action.preflight_command = value;
    } else if (key == "network_interface" || key == "net") {
        action.network_interface = value;
    } else if (key == "duration_sec") {
        action.duration_sec = std::stod(value);
    } else if (key == "max_frames" || key == "max_steps" ||
               key == "motion_end_step" || key == "motion_end_frame") {
        action.max_frames = std::stoi(value);
    } else if (key == "switch_back") {
        action.switch_back = parseBool(value, true);
    } else if (key == "stop_loco_before_start") {
        action.stop_loco_before_start = parseBool(value, true);
    } else if (key == "safety_enable" || key == "safety" ||
               key == "enable_safety" || key == "safety_mode") {
        action.safety_enable = parseBool(value, true);
    } else if (key == "prestart_before_userctrl" || key == "start_before_userctrl") {
        action.prestart_before_userctrl = parseBool(value, false);
    } else if (key == "prestart_ready_timeout_sec" || key == "ready_timeout_sec") {
        action.prestart_ready_timeout_sec = std::stod(value);
    } else if (key == "ready_file") {
        action.ready_file = value;
    } else if (key == "start_file") {
        action.start_file = value;
    } else if (key == "postrun_handoff_hold") {
        action.postrun_handoff_hold = parseBool(value, false);
    } else if (key == "handoff_complete_file") {
        action.handoff_complete_file = value;
    } else if (key == "handoff_release_file") {
        action.handoff_release_file = value;
    } else if (key == "switch_back_overlap_sec") {
        action.switch_back_overlap_sec = std::stod(value);
        if (action.switch_back_overlap_sec < 0.0) {
            throw std::runtime_error("switch_back_overlap_sec must be >= 0");
        }
    } else if (key == "switch_back_mode" || key == "internal_ctrl_mode" || key == "back_mode") {
        action.switch_back_mode = normalizeInternalMode(value);
    } else {
        throw std::runtime_error("unknown motion action field: " + key);
    }
}

struct ConfigDefaults
{
    MotionActionController::ActionConfig config;
    std::unordered_set<std::string> keys;
};

void setDefaultField(ConfigDefaults& defaults,
                     const std::string& key,
                     const std::string& value)
{
    applyActionField(defaults.config, key, value);
    defaults.keys.insert(key);
}

void applyDefaults(MotionActionController::ActionConfig& action,
                   const ConfigDefaults& defaults)
{
    for (const auto& key : defaults.keys) {
        if (key == "type") {
            action.type = defaults.config.type;
        } else if (key == "model") {
            action.model = defaults.config.model;
        } else if (key == "params") {
            action.params = defaults.config.params;
        } else if (key == "trajectory") {
            action.trajectory = defaults.config.trajectory;
        } else if (key == "command") {
            action.command = defaults.config.command;
        } else if (key == "preflight_command" || key == "dry_run_command") {
            action.preflight_command = defaults.config.preflight_command;
        } else if (key == "network_interface" || key == "net") {
            action.network_interface = defaults.config.network_interface;
        } else if (key == "duration_sec") {
            action.duration_sec = defaults.config.duration_sec;
        } else if (key == "max_frames" || key == "max_steps" ||
                   key == "motion_end_step" || key == "motion_end_frame") {
            action.max_frames = defaults.config.max_frames;
        } else if (key == "switch_back") {
            action.switch_back = defaults.config.switch_back;
        } else if (key == "stop_loco_before_start") {
            action.stop_loco_before_start = defaults.config.stop_loco_before_start;
        } else if (key == "safety_enable" || key == "safety" ||
                   key == "enable_safety" || key == "safety_mode") {
            action.safety_enable = defaults.config.safety_enable;
        } else if (key == "prestart_before_userctrl" || key == "start_before_userctrl") {
            action.prestart_before_userctrl = defaults.config.prestart_before_userctrl;
        } else if (key == "prestart_ready_timeout_sec" || key == "ready_timeout_sec") {
            action.prestart_ready_timeout_sec = defaults.config.prestart_ready_timeout_sec;
        } else if (key == "ready_file") {
            action.ready_file = defaults.config.ready_file;
        } else if (key == "start_file") {
            action.start_file = defaults.config.start_file;
        } else if (key == "postrun_handoff_hold") {
            action.postrun_handoff_hold = defaults.config.postrun_handoff_hold;
        } else if (key == "handoff_complete_file") {
            action.handoff_complete_file = defaults.config.handoff_complete_file;
        } else if (key == "handoff_release_file") {
            action.handoff_release_file = defaults.config.handoff_release_file;
        } else if (key == "switch_back_overlap_sec") {
            action.switch_back_overlap_sec = defaults.config.switch_back_overlap_sec;
        } else if (key == "switch_back_mode" || key == "internal_ctrl_mode" || key == "back_mode") {
            action.switch_back_mode = defaults.config.switch_back_mode;
        }
    }
}

int internalModeId(const std::string& mode)
{
    const std::string normalized = normalizeInternalMode(mode);
    if (normalized == "last") {
        return 0;
    }
    if (normalized == "passive") {
        return 1;
    }
    return 2; // walkrun
}

const char* internalModeName(int mode_id)
{
    switch (mode_id) {
    case 0: return "LAST";
    case 1: return "PASSIVE";
    case 2: return "WALKRUN";
    default: return "UNKNOWN";
    }
}

bool serviceSwitchAiSport(int enable, std::string& error_message)
{
    try {
        unitree::robot::b2::RobotStateClient robot_state_client;
        robot_state_client.Init();
        robot_state_client.SetTimeout(5.0f);
        int32_t status = 0;
        const int32_t ret = robot_state_client.ServiceSwitch("ai_sport", enable, status);
        if (ret != 0) {
            error_message = "ServiceSwitch(ai_sport," + std::to_string(enable) +
                            ") failed ret=" + std::to_string(ret);
            return false;
        }
        if (enable == 0 && status == 0) {
            error_message = "ServiceSwitch(ai_sport,0) returned status=0, service still enabled";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        error_message = std::string("ServiceSwitch(ai_sport,") + std::to_string(enable) +
                        ") exception: " + e.what();
        return false;
    }
}

uint32_t prearmCrc32(uint32_t* ptr, uint32_t len)
{
    uint32_t crc = 0xFFFFFFFF;
    constexpr uint32_t polynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t bit = 1u << 31;
        const uint32_t data = ptr[i];
        for (uint32_t j = 0; j < 32; ++j) {
            if (crc & 0x80000000) {
                crc <<= 1;
                crc ^= polynomial;
            } else {
                crc <<= 1;
            }
            if (data & bit) crc ^= polynomial;
            bit >>= 1;
        }
    }
    return crc;
}

bool publishUserCtrlPrearm(std::string& error_message)
{
    // Unitree's g1_userctrl_dds_example.cpp publishes one harmless
    // rt/user_lowcmd packet before SwitchToUserCtrl().  This makes sure the
    // user-control command channel already exists when the official controller
    // hands control over.
    try {
        unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_> publisher(kUserLowCmdTopic);
        publisher.InitChannel();

        const auto prearm = action_go::runtime::validated_prearm_command_values();

        unitree_hg::msg::dds_::LowCmd_ msg;
        msg.mode_pr() = 0;
        // The G1 protocol's normal mode_machine value is 5.  The active
        // LowState value is supplied by the policy publisher after handoff.
        msg.mode_machine() = 5;
        const size_t count = std::min<size_t>(kG1MotorCount, msg.motor_cmd().size());
        for (size_t i = 0; i < count; ++i) {
            msg.motor_cmd().at(i).mode() = 1;
            msg.motor_cmd().at(i).q(prearm.position);
            msg.motor_cmd().at(i).dq(prearm.velocity);
            msg.motor_cmd().at(i).kp(prearm.kp);
            msg.motor_cmd().at(i).kd(prearm.kd);
            msg.motor_cmd().at(i).tau(prearm.torque);
        }
        msg.crc() = prearmCrc32(reinterpret_cast<uint32_t*>(&msg),
                                (sizeof(msg) >> 2) - 1);
        publisher.Write(msg);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return true;
    } catch (const std::exception& e) {
        error_message = std::string("user_lowcmd prearm failed: ") + e.what();
        return false;
    }
}

class LocoUserCtrlCompatClient : public unitree::robot::g1::LocoClient
{
public:
    void Init()
    {
        unitree::robot::g1::LocoClient::Init();
        // Newer G1 SDK headers define these as:
        // ROBOT_API_ID_LOCO_SWITCH_TO_USER_CTRL     = 7110
        // ROBOT_API_ID_LOCO_SWITCH_TO_INTERNAL_CTRL = 7111
        // Register them manually so older headers can still call newer robot
        // firmware without falling back to ai_sport ServiceSwitch.
        RegistApi(kSwitchToUserCtrlApiId, 0);
        RegistApi(kSwitchToInternalCtrlApiId, 0);
    }

    int32_t SwitchToUserCtrlCompat()
    {
        std::string parameter;
        std::string data;
        unitree::robot::go2::JsonizeDataBool json;
        parameter = unitree::common::ToJsonString(json);
        return Call(kSwitchToUserCtrlApiId, parameter, data);
    }

    int32_t SwitchToInternalCompat(int mode_id)
    {
        std::string parameter;
        std::string data;
        unitree::robot::go2::JsonizeDataInt json;
        json.data = mode_id; // LAST=0, PASSIVE=1, WALKRUN=2
        parameter = unitree::common::ToJsonString(json);
        return Call(kSwitchToInternalCtrlApiId, parameter, data);
    }

private:
    static constexpr int32_t kSwitchToUserCtrlApiId = 7110;
    static constexpr int32_t kSwitchToInternalCtrlApiId = 7111;
};

bool callRawSwitchToUserCtrlApi(std::string& error_message)
{
    try {
        LocoUserCtrlCompatClient client;
        client.Init();
        client.SetTimeout(5.0f);
        const int32_t ret = client.SwitchToUserCtrlCompat();
        if (ret != 0) {
            error_message = "raw loco API 7110 SwitchToUserCtrl failed ret=" + std::to_string(ret);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        error_message = std::string("raw loco API 7110 exception: ") + e.what();
        return false;
    }
}

bool callRawSwitchToInternalApi(int mode_id, std::string& error_message)
{
    try {
        LocoUserCtrlCompatClient client;
        client.Init();
        client.SetTimeout(5.0f);
        const int32_t ret = client.SwitchToInternalCompat(mode_id);
        if (ret != 0) {
            error_message = "raw loco API 7111 SwitchToInternalCtrl(" +
                            std::to_string(mode_id) + ") failed ret=" + std::to_string(ret);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        error_message = std::string("raw loco API 7111 exception: ") + e.what();
        return false;
    }
}

class UnitreeControlModeBackend final : public action_go::runtime::ControlModeBackend
{
public:
    explicit UnitreeControlModeBackend(unitree::robot::g1::LocoClient* client)
        : client_(client)
    {
    }

    const char* name() const override
    {
        return "unitree_g1_loco";
    }

    bool stop_move(std::string& error_message) override
    {
        if (!client_) {
            error_message = "LocoClient is null";
            return false;
        }
        const int32_t ret = client_->StopMove();
        if (ret != 0) {
            error_message = "LocoClient::StopMove failed ret=" + std::to_string(ret);
            return false;
        }
        return true;
    }

    bool switch_to_user_ctrl(bool publish_prearm,
                             std::string& error_message) override
    {
        if (!client_) {
            error_message = "LocoClient is null";
            return false;
        }

        int fsm_id = -1;
        const int32_t get_ret = client_->GetFsmId(fsm_id);
        if (get_ret == 0) {
            std::cout << "[MotionActionController] current official fsm_id before user ctrl: "
                      << fsm_id << std::endl;
        }

        if (publish_prearm) {
            if (!publishUserCtrlPrearm(error_message)) {
                return false;
            }
        }

        std::string native_error;
        if (callNativeSwitchToUserCtrl(client_, native_error)) {
            std::cout << "[MotionActionController] switched to user ctrl by "
                         "LocoClient::SwitchToUserCtrl()" << std::endl;
            return true;
        }

        std::cerr << "[MotionActionController] native SwitchToUserCtrl unavailable/failed: "
                  << native_error << std::endl;

        std::string raw_error;
        if (callRawSwitchToUserCtrlApi(raw_error)) {
            std::cout << "[MotionActionController] switched to user ctrl by raw loco API 7110"
                      << std::endl;
            return true;
        }

        std::cerr << "[MotionActionController] raw loco API 7110 failed, "
                     "fallback to ServiceSwitch: " << raw_error << std::endl;
        if (serviceSwitchAiSport(0, error_message)) {
            return true;
        }
        error_message = native_error + "; " + raw_error + "; fallback " + error_message;
        return false;
    }

    bool switch_to_internal(action_go::runtime::OfficialControlMode mode,
                            std::string& error_message) override
    {
        if (!client_) {
            error_message = "LocoClient is null";
            return false;
        }

        const int mode_id = static_cast<int>(mode);
        const char* mode_name = internalModeName(mode_id);
        std::string native_error;
        if (callNativeSwitchToInternal(client_, mode_id, native_error)) {
            std::cout << "[MotionActionController] switched back by "
                         "LocoClient::SwitchToInternalCtrl(" << mode_name << ")" << std::endl;
            return true;
        }

        std::cerr << "[MotionActionController] native SwitchToInternalCtrl unavailable/failed: "
                  << native_error << std::endl;

        std::string raw_error;
        if (callRawSwitchToInternalApi(mode_id, raw_error)) {
            std::cout << "[MotionActionController] switched back by raw loco API 7111 "
                      << mode_name << std::endl;
            return true;
        }

        std::cerr << "[MotionActionController] raw loco API 7111 failed, "
                     "fallback to ServiceSwitch: " << raw_error << std::endl;
        if (serviceSwitchAiSport(1, error_message)) {
            return true;
        }
        error_message = native_error + "; " + raw_error + "; fallback " + error_message;
        return false;
    }

    bool enter_damping(std::string& error_message) override
    {
        if (!client_) {
            error_message = "LocoClient is null";
            return false;
        }

        int32_t stop_ret = client_->StopMove();
        int32_t damp_ret = -1;
        for (int i = 0; i < 3; ++i) {
            damp_ret = client_->Damp();
            if (damp_ret == 0) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        std::string restore_error;
        if (!switch_to_internal(action_go::runtime::OfficialControlMode::kPassive,
                                restore_error)) {
            std::cerr << "[MotionActionController] enterOfficialDamping restore warning: "
                      << restore_error << std::endl;
        }

        stop_ret = client_->StopMove();
        for (int i = 0; i < 5; ++i) {
            damp_ret = client_->Damp();
            if (damp_ret == 0) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        error_message = "StopMove ret=" + std::to_string(stop_ret) +
                        ", Damp ret=" + std::to_string(damp_ret);
        if (!restore_error.empty()) {
            error_message += ", restore: " + restore_error;
        }
        return false;
    }

private:
    unitree::robot::g1::LocoClient* client_{nullptr};
};
} // namespace

namespace action_go::runtime {

const char* official_control_mode_name(OfficialControlMode mode)
{
    switch (mode) {
    case OfficialControlMode::kLast: return "last";
    case OfficialControlMode::kPassive: return "passive";
    case OfficialControlMode::kWalkRun: return "walkrun";
    }
    return "walkrun";
}

}  // namespace action_go::runtime

MotionActionController::MotionActionController()
    : supervisor_platform_(action_go::runtime::make_posix_supervisor_platform())
{
}

MotionActionController::~MotionActionController()
{
    cancel();
}

void MotionActionController::setEventLogPath(const std::string& path)
{
    event_log_path_ = trim(path);
    event_log_.open(event_log_path_);
    base_event_context_ = event_log_.context();
    if (!event_log_path_.empty()) {
        event_log_.log("supervisor_log_opened",
                       {event_string("path", event_log_path_)});
    }
}

bool MotionActionController::init(unitree::robot::g1::LocoClient* loco_client,
                                  const std::string& config_path,
                                  const std::string& network_interface_override,
                                  bool dry_run)
{
    if (!loco_client) {
        setFailed("MotionActionController init failed: loco_client is null");
        return false;
    }
    return init(std::make_shared<UnitreeControlModeBackend>(loco_client), config_path,
                network_interface_override, dry_run);
}

bool MotionActionController::init(
    std::shared_ptr<action_go::runtime::ControlModeBackend> control_backend,
    const std::string& config_path,
    const std::string& network_interface_override,
    bool dry_run)
{
    return init(std::move(control_backend),
                action_go::runtime::make_posix_supervisor_platform(), config_path,
                network_interface_override, dry_run);
}

bool MotionActionController::init(
    std::shared_ptr<action_go::runtime::ControlModeBackend> control_backend,
    std::shared_ptr<action_go::runtime::SupervisorPlatform> supervisor_platform,
    const std::string& config_path,
    const std::string& network_interface_override,
    bool dry_run)
{
    control_backend_ = std::move(control_backend);
    supervisor_platform_ = std::move(supervisor_platform);
    network_interface_override_ = trim(network_interface_override);
    dry_run_ = dry_run;
    if (!control_backend_) {
        setFailed("MotionActionController init failed: control backend is null");
        return false;
    }
    if (!supervisor_platform_) {
        setFailed("MotionActionController init failed: supervisor platform is null");
        return false;
    }
    if (!loadConfig(config_path)) {
        return false;
    }
    event_log_.log("motion_config_loaded",
                   {event_string("config", fs::absolute(config_path).lexically_normal().string()),
                    event_string("control_backend", control_backend_->name()),
                    event_int("action_count", static_cast<long long>(actions_.size())),
                    event_bool("dry_run", dry_run_)});
    setStatus(Status::Idle);
    return true;
}

bool MotionActionController::loadConfig(const std::string& config_path)
{
    const fs::path main_config_file = fs::absolute(config_path).lexically_normal();

    config_dir_ = main_config_file.parent_path();
    // motion_actions.yaml lives under action_go/config by convention.
    action_go_root_ = config_dir_.filename() == "config" ? config_dir_.parent_path() : config_dir_;

    actions_.clear();
    ConfigDefaults global_defaults;
    std::unordered_map<std::string, ConfigDefaults> type_defaults;
    std::unordered_set<std::string> loaded_config_files;

    auto resolveIncludedActionFile = [&](const fs::path& including_dir,
                                         const std::string& raw_path) -> fs::path {
        fs::path path(raw_path);
        if (path.is_absolute()) {
            return path.lexically_normal();
        }

        const fs::path from_including_dir = (including_dir / path).lexically_normal();
        if (fs::exists(from_including_dir)) {
            return from_including_dir;
        }

        const fs::path from_root = (action_go_root_ / path).lexically_normal();
        if (fs::exists(from_root)) {
            return from_root;
        }

        const fs::path from_config_dir = (config_dir_ / path).lexically_normal();
        if (fs::exists(from_config_dir)) {
            return from_config_dir;
        }

        // Prefer action_go-root-relative paths for stable error messages.
        return from_root;
    };

    enum class ParseSection { None, Defaults, TypeDefaults, Actions, ActionFiles };

    std::function<bool(const fs::path&)> parseConfigFile = [&](const fs::path& raw_file) -> bool {
        const fs::path config_file = fs::absolute(raw_file).lexically_normal();
        const std::string config_file_key = config_file.string();
        if (!loaded_config_files.insert(config_file_key).second) {
            return true;
        }

        std::ifstream input(config_file);
        if (!input.good()) {
            setFailed("failed to open motion action config/action file: " + config_file.string());
            return false;
        }

        ParseSection section = ParseSection::None;
        ActionConfig* current = nullptr;
        ConfigDefaults* current_type_defaults = nullptr;
        std::string current_action_group_type;
        std::vector<fs::path> pending_action_files;

        std::string line;
        while (std::getline(input, line)) {
            const auto hash_pos = line.find('#');
            if (hash_pos != std::string::npos) {
                line = line.substr(0, hash_pos);
            }
            if (trim(line).empty()) {
                continue;
            }

            const auto indent = line.find_first_not_of(' ');
            const std::string stripped = trim(line);

            if (section == ParseSection::ActionFiles && indent == 2 &&
                !stripped.empty() && stripped.front() == '-') {
                std::string include_path = unquote(trim(stripped.substr(1)));
                if (!include_path.empty()) {
                    pending_action_files.push_back(
                        resolveIncludedActionFile(config_file.parent_path(), include_path));
                }
                continue;
            }

            const auto colon = stripped.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error("invalid motion action YAML line (missing ':') in " +
                                         config_file.string() + ": " + stripped);
            }

            const std::string key = trim(stripped.substr(0, colon));
            const std::string value = unquote(stripped.substr(colon + 1));

            if (indent == 0 && value.empty()) {
                current = nullptr;
                current_type_defaults = nullptr;
                current_action_group_type.clear();
                if (key == "defaults") {
                    section = ParseSection::Defaults;
                } else if (key == "type_defaults") {
                    section = ParseSection::TypeDefaults;
                } else if (key == "actions") {
                    section = ParseSection::Actions;
                } else if (key == "action_files" || key == "actions_files" ||
                           key == "include_actions" || key == "includes") {
                    section = ParseSection::ActionFiles;
                } else {
                    throw std::runtime_error("unknown motion action top-level section: " + key);
                }
                continue;
            }

            if (section == ParseSection::ActionFiles && indent == 2) {
                const std::string include_path = value.empty() ? key : value;
                if (!include_path.empty()) {
                    pending_action_files.push_back(
                        resolveIncludedActionFile(config_file.parent_path(), include_path));
                }
                continue;
            }

            if (section == ParseSection::Defaults && indent == 2) {
                setDefaultField(global_defaults, key, value);
                continue;
            }

            if (section == ParseSection::TypeDefaults) {
                if (indent == 2 && value.empty()) {
                    current_type_defaults = &type_defaults[key];
                    continue;
                }
                if (current_type_defaults && indent == 4) {
                    setDefaultField(*current_type_defaults, key, value);
                    continue;
                }
            }

            if (section == ParseSection::Actions) {
                if (indent == 2 && value.empty()) {
                    const bool is_group = type_defaults.find(key) != type_defaults.end();
                    if (is_group) {
                        current_action_group_type = key;
                        current = nullptr;
                        continue;
                    }

                    current_action_group_type.clear();
                    auto inserted = actions_.emplace(key, ActionConfig{});
                    if (!inserted.second) {
                        throw std::runtime_error("duplicate motion action id: " + key);
                    }
                    inserted.first->second = ActionConfig{};
                    inserted.first->second.name = key;
                    inserted.first->second.resource_base_dir = config_file.parent_path().string();
                    applyDefaults(inserted.first->second, global_defaults);
                    current = &inserted.first->second;
                    continue;
                }

                if (!current_action_group_type.empty() && indent == 4 && value.empty()) {
                    auto inserted = actions_.emplace(key, ActionConfig{});
                    if (!inserted.second) {
                        throw std::runtime_error("duplicate motion action id: " + key);
                    }
                    inserted.first->second = ActionConfig{};
                    inserted.first->second.name = key;
                    inserted.first->second.resource_base_dir = config_file.parent_path().string();
                    applyDefaults(inserted.first->second, global_defaults);
                    inserted.first->second.type = current_action_group_type;
                    const auto defaults_it = type_defaults.find(current_action_group_type);
                    if (defaults_it != type_defaults.end()) {
                        const std::string name = inserted.first->second.name;
                        const std::string type = inserted.first->second.type;
                        applyDefaults(inserted.first->second, defaults_it->second);
                        inserted.first->second.name = name;
                        inserted.first->second.type = type;
                    }
                    current = &inserted.first->second;
                    continue;
                }

                const size_t field_indent = current_action_group_type.empty() ? 4 : 6;
                if (!current || indent < field_indent) {
                    continue;
                }

                applyActionField(*current, key, value);
                if (key == "type") {
                    const auto defaults_it = type_defaults.find(current->type);
                    if (defaults_it != type_defaults.end()) {
                        const std::string name = current->name;
                        const std::string type = current->type;
                        applyDefaults(*current, defaults_it->second);
                        current->name = name;
                        current->type = type;
                    }
                }
            }
        }

        for (const auto& action_file : pending_action_files) {
            if (!parseConfigFile(action_file)) {
                return false;
            }
        }
        return true;
    };

    if (!parseConfigFile(main_config_file)) {
        return false;
    }

    for (auto& [name, action] : actions_) {
        (void)name;
        if (!network_interface_override_.empty()) {
            action.network_interface = network_interface_override_;
        }
        action.model = resolveResourcePath(action.model, action.resource_base_dir);
        action.params = resolveResourcePath(action.params, action.resource_base_dir);
        // BeyondMimic trajectory is the max_time_step scalar.  Keep numeric
        // values as-is so legacy configs can pass "trajectory: 500" directly
        // to the player instead of resolving it as a txt file path.
        if (!(action.type == "beyondmimic" && isNumericLiteral(action.trajectory))) {
            action.trajectory = resolveResourcePath(action.trajectory, action.resource_base_dir);
        }
        if (action.command.empty() && action.type == "beyondmimic" &&
            !action.model.empty() && !action.params.empty() && !action.trajectory.empty()) {
            action.command = "{player} --net {net} --cmd-topic rt/user_lowcmd --model {model} --params {params} --trajectory {trajectory}";
        }
        if (action.prestart_before_userctrl) {
            if (action.ready_file.empty()) {
                action.ready_file = "/tmp/action_go_" + action.name + "_ready";
            }
            if (action.start_file.empty()) {
                action.start_file = "/tmp/action_go_" + action.name + "_start";
            }
        }
        if (action.postrun_handoff_hold) {
            if (action.handoff_complete_file.empty()) {
                action.handoff_complete_file = "/tmp/action_go_" + action.name + "_complete";
            }
            if (action.handoff_release_file.empty()) {
                action.handoff_release_file = "/tmp/action_go_" + action.name + "_release";
            }
        }
    }

    if (actions_.empty()) {
        setFailed("motion action config contains no actions: " + config_path);
        return false;
    }

    std::cout << "[MotionActionController] loaded " << actions_.size()
              << " actions from " << main_config_file.string()
              << " action_go_root=" << action_go_root_.string();
    if (!network_interface_override_.empty()) {
        std::cout << " net_override=" << network_interface_override_;
    }
    if (dry_run_) {
        std::cout << " DRY_RUN=1";
    }
    std::cout << std::endl;
    return true;
}

void MotionActionController::prepareActionEventLog(ActionConfig& action)
{
    action.supervisor_event_log = event_log_path_;
    action.event_log_arg.clear();
    action.payload_event_log.clear();
    action.action_seq = ++next_action_seq_;
    auto context = base_event_context_;
    const std::string trial_prefix = context.trial_id.empty() || context.trial_id == "auto"
        ? (context.run_id.empty() ? action.name : context.run_id)
        : context.trial_id;
    context.trial_id = trial_prefix + "_" + std::to_string(action.action_seq);
    if (context.policy.empty() || context.policy == "auto") {
        context.policy = action.name;
    }
    if (context.handoff_method.empty() || context.handoff_method == "auto") {
        context.handoff_method = action.prestart_before_userctrl
            ? "hot_standby"
            : "cold_start";
    }
    action.trial_id = context.trial_id;
    event_log_.set_context(std::move(context));

    if (event_log_path_.empty()) {
        return;
    }

    fs::path supervisor_path(event_log_path_);
    fs::path dir = supervisor_path.has_parent_path() ? supervisor_path.parent_path() : fs::current_path();
    const std::string stem = supervisor_path.stem().empty() ? "actiongo" : supervisor_path.stem().string();
    const std::string name = safeFileComponent(action.name);
    action.payload_event_log =
        (dir / (stem + "_payload_" + std::to_string(action.action_seq) + "_" + name + ".jsonl")).string();
    action.event_log_arg = "--event-log " + shellQuote(action.payload_event_log);
}

void MotionActionController::logActionEvent(const std::string& event,
                                            const ActionConfig& action,
                                            const std::string& reason)
{
    event_log_.log(event,
                   {event_string("action", action.name),
                    event_string("type", action.type),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_string("ready_file", action.ready_file),
                    event_string("start_file", action.start_file),
                    event_string("handoff_complete_file", action.handoff_complete_file),
                    event_string("handoff_release_file", action.handoff_release_file),
                    event_string("payload_event_log", action.payload_event_log),
                    event_string("reason", reason)});
}

bool MotionActionController::startAction(const std::string& action_name)
{
    ActionConfig action;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const Status current_status = status_.load();
        if (current_status == Status::SwitchingToUserCtrl ||
            current_status == Status::Running ||
            current_status == Status::SwitchingBack ||
            current_status == Status::SwitchBackBlocked) {
            last_error_ = "another motion action is running: " + current_action_;
            return false;
        }

        const auto it = actions_.find(action_name);
        if (it == actions_.end()) {
            last_error_ = "unknown motion action: " + action_name;
            status_.store(Status::Failed);
            return false;
        }
        action = it->second;
        prepareActionEventLog(action);
    }

    joinWorkerIfJoinable();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_action_ = action_name;
        current_action_seq_ = action.action_seq;
        current_trial_id_ = action.trial_id;
        current_event_context_ = event_log_.context();
        current_switch_back_mode_ = action.switch_back_mode;
        last_error_.clear();
        cancel_requested_.store(false);
        suppress_switch_back_on_cancel_.store(false);
        handoff_release_sent_.store(false);
        status_.store(Status::SwitchingToUserCtrl);
    }
    logActionEvent("action_triggered", action,
                   action.prestart_before_userctrl ? "prestart" : "direct");
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        worker_ = std::thread(&MotionActionController::workerMain, this, action);
    }
    return true;
}

MotionActionController::Status MotionActionController::update()
{
    const Status s = status_.load();
    if (s == Status::Done || s == Status::Failed || s == Status::Idle) {
        joinWorkerIfJoinable();
    }
    return s;
}

void MotionActionController::cancel()
{
    suppress_switch_back_on_cancel_.store(false);
    cancel_requested_.store(true);
    terminateRunningProcess(false);
    joinWorkerIfJoinable();
    if (status_.load() != Status::Failed) {
        setStatus(Status::Idle);
    }
}

void MotionActionController::interruptForRecovery()
{
    std::cerr << "[MotionActionController] safety recovery interrupt requested" << std::endl;
    if (dry_run_) {
        cancel_requested_.store(true);
        terminateRunningProcess(false);
        joinWorkerIfJoinable();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_error_.clear();
            current_action_.clear();
        }
        setStatus(Status::Idle);
        return;
    }

    // Unlike dampingStop()/stopAndSwitchToOfficial(), keep the robot out of
    // the official controller hand-back path.  The caller will immediately
    // start a recovery action (BFM-Zero) in UserCtrl.
    suppress_switch_back_on_cancel_.store(true);
    cancel_requested_.store(true);
    terminateRunningProcess(false);
    joinWorkerIfJoinable();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_.clear();
        current_action_.clear();
    }
    setStatus(Status::Idle);
    suppress_switch_back_on_cancel_.store(false);
}

void MotionActionController::dampingStop()
{
    std::cerr << "[MotionActionController] dampingStop requested" << std::endl;
    if (dry_run_) {
        cancel_requested_.store(true);
        terminateRunningProcess(true);
        joinWorkerIfJoinable();
        setFailed("dry-run damping stopped");
        return;
    }
    // We are about to force official damping.  Suppress the worker thread's
    // normal action.switch_back path; otherwise a canceled action whose
    // switch_back_mode is walkrun can race with dampingStop() and switch the
    // robot back to WALKRUN immediately after we requested damping.
    suppress_switch_back_on_cancel_.store(true);
    cancel_requested_.store(true);

    terminateRunningProcess(true);

    std::string damping_error;
    if (!enterOfficialDamping(damping_error)) {
        std::cerr << "[MotionActionController] dampingStop warning: "
                  << damping_error << std::endl;
    } else {
        std::string action;
        uint64_t action_seq = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            action = current_action_;
            action_seq = current_action_seq_;
        }
        event_log_.log("fallback_completed",
                       {event_string("action", action),
                        event_int("action_seq", static_cast<long long>(action_seq)),
                        event_string("mode", "passive"),
                        event_string("reason", "damping_stop")});
    }

    joinWorkerIfJoinable();
    suppress_switch_back_on_cancel_.store(false);
    setFailed("motion action damping stopped");
}

bool MotionActionController::stopAndSwitchToOfficial(const std::string& mode)
{
    const std::string normalized_mode = normalizeInternalMode(mode);
    std::cerr << "[MotionActionController] normal stop requested, switch to official "
              << normalized_mode << std::endl;
    if (dry_run_) {
        cancel_requested_.store(true);
        terminateRunningProcess(false);
        joinWorkerIfJoinable();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_error_.clear();
            current_action_.clear();
            current_switch_back_mode_ = normalized_mode;
        }
        setStatus(Status::Idle);
        std::cerr << "[MotionActionController][dry-run] skip official switch to "
                  << normalized_mode << std::endl;
        return true;
    }

    const Status current_status = status_.load();
    const bool cancel_before_rpc = current_status == Status::SwitchingToUserCtrl;
    ActionConfig action = currentActionConfig();
    action.switch_back_mode = normalized_mode;
    // A stop racing the UserCtrl RPC still cancels early.  Once UserCtrl is
    // stable, the payload must continue publishing through the official RPC
    // and the configured overlap window.
    suppress_switch_back_on_cancel_.store(true);
    if (cancel_before_rpc) cancel_requested_.store(true);
    std::string switch_error;
    const bool ok = completeSwitchBack(action, normalized_mode,
                                       "external_normal_stop", switch_error);
    if (!ok) {
        // The payload remains active and can be retried.  This is deliberately
        // not Failed: dampingStop must remain available.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_error_ = "normal stop switch back failed: " + switch_error;
        }
        setStatus(Status::SwitchBackBlocked);
        return false;
    }

    const bool releasing_completed_hold = action.postrun_handoff_hold &&
        supervisor_platform_ &&
        supervisor_platform_->file_exists(action.handoff_complete_file);
    if (!releasing_completed_hold) {
        cancel_requested_.store(true);
        terminateRunningProcess(false);
    }
    joinWorkerIfJoinable();
    suppress_switch_back_on_cancel_.store(false);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_.clear();
        current_action_.clear();
        current_switch_back_mode_ = normalized_mode;
    }
    setStatus(Status::Idle);
    return true;
}

MotionActionController::ActionConfig MotionActionController::currentActionConfig() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = actions_.find(current_action_);
    ActionConfig action = it == actions_.end() ? ActionConfig{} : it->second;
    action.name = current_action_;
    action.action_seq = current_action_seq_;
    action.trial_id = current_trial_id_;
    return action;
}

bool MotionActionController::isRunning() const
{
    const Status s = status_.load();
    return s == Status::SwitchingToUserCtrl || s == Status::Running ||
           s == Status::SwitchingBack || s == Status::SwitchBackBlocked;
}

bool MotionActionController::isDone() const
{
    return status_.load() == Status::Done;
}

bool MotionActionController::hasError() const
{
    return status_.load() == Status::Failed;
}

std::string MotionActionController::lastError() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
}

std::string MotionActionController::currentAction() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return current_action_;
}

bool MotionActionController::actionSafetyEnabled(const std::string& action_name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = actions_.find(action_name);
    if (it == actions_.end()) {
        // Unknown actions will fail to start; keep safety conservative for any
        // caller-side race or stale name.
        return true;
    }
    return it->second.safety_enable;
}

bool MotionActionController::hasAction(const std::string& action_name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return actions_.find(action_name) != actions_.end();
}

const char* MotionActionController::statusToString(Status status)
{
    switch (status) {
    case Status::Idle: return "Idle";
    case Status::SwitchingToUserCtrl: return "SwitchingToUserCtrl";
    case Status::Running: return "Running";
    case Status::SwitchingBack: return "SwitchingBack";
    case Status::SwitchBackBlocked: return "SwitchBackBlocked";
    case Status::Done: return "Done";
    case Status::Failed: return "Failed";
    }
    return "Unknown";
}

void MotionActionController::workerMain(ActionConfig action)
{
    std::string error;
    bool payload_prestarted = false;
    bool loco_stopped = false;

    auto clear_handoff_files = [&]() {
        if (!supervisor_platform_) {
            error = "supervisor platform is null";
            return false;
        }
        if (!supervisor_platform_->remove_file(action.ready_file, error)) return false;
        if (!supervisor_platform_->remove_file(action.start_file, error)) return false;
        if (!supervisor_platform_->remove_file(action.handoff_complete_file, error)) return false;
        if (!supervisor_platform_->remove_file(action.handoff_release_file, error)) return false;
        return true;
    };

    auto request_switch_back_after_pre_userctrl_failure = [&]() {
        if (dry_run_) {
            std::cerr << "[MotionActionController][dry-run] pre-UserCtrl failure; "
                      << "skip official switch back to "
                      << normalizeInternalMode(action.switch_back_mode) << std::endl;
            return;
        }
        std::string switch_error;
        if (!switchBackToOfficialLogged(switch_error, action.switch_back_mode,
                                        action, "pre_userctrl_failure")) {
            std::cerr << "[MotionActionController] pre-UserCtrl failure; switch back warning: "
                      << switch_error << std::endl;
        }
    };

    if (!action.preflight_command.empty()) {
        if (cancel_requested_.load()) {
            setStatus(Status::Idle);
            return;
        }
        const std::string preflight = expandCommandTemplate(action, action.preflight_command);
        std::cout << "[MotionActionController] running preflight dry-run before UserCtrl: "
                  << preflight << std::endl;
        if (!runCommandString(preflight, error)) {
            if (cancel_requested_.load()) {
                setStatus(Status::Idle);
                return;
            }
            request_switch_back_after_pre_userctrl_failure();
            setFailed("preflight dry-run failed; real LowCmd was not started; requested official " +
                      normalizeInternalMode(action.switch_back_mode) + ": " + error);
            return;
        }
        std::cout << "[MotionActionController] preflight dry-run OK"
                  << (dry_run_ ? ", staying in dry-run mode" : ", switching to UserCtrl")
                  << std::endl;
    }

    if (dry_run_) {
        const std::string dry_command = dryRunCommandString(action);
        std::cout << "[MotionActionController][dry-run] no StopMove/UserCtrl/official switch; "
                  << "execute player dry-run command: " << dry_command << std::endl;
        setStatus(Status::Running);

        bool payload_ok = false;
        if (action.prestart_before_userctrl && action.type != "instinct_parkour" &&
            !action.ready_file.empty() && !action.start_file.empty()) {
            if (!clear_handoff_files()) {
                setFailed("dry-run failed to clear handoff files: " + error);
                return;
            }
            if (!startCommandStringAsync(dry_command, error)) {
                setFailed("dry-run prestart failed: " + error);
                return;
            }
            event_log_.log("payload_spawned",
                           {event_string("action", action.name),
                            event_int("action_seq", static_cast<long long>(action.action_seq)),
                            event_int("pid", static_cast<long long>(running_pid_.load())),
                            event_string("phase", "dry_run_prestart"),
                            event_string("payload_event_log", action.payload_event_log)});
            if (!waitForFile(action.ready_file, action.prestart_ready_timeout_sec, error)) {
                terminateRunningProcess(false);
                setFailed("dry-run ready timeout: " + error);
                return;
            }
            logActionEvent("ready_seen", action, "dry_run");
            if (!touchFile(action.start_file, error)) {
                terminateRunningProcess(false);
                setFailed("dry-run failed to signal start file: " + error);
                return;
            }
            logActionEvent("start_file_touched", action, "dry_run");
            payload_ok = waitRunningCommand(dry_command, error, &action);
        } else {
            payload_ok = runCommandString(action, dry_command, error, "dry_run_payload");
        }

        if (!payload_ok) {
            if (cancel_requested_.load()) {
                logActionEvent("action_exit", action, "canceled");
                setStatus(Status::Idle);
                return;
            }
            logActionEvent("action_exit", action, error);
            setFailed("dry-run action command failed: " + error);
            return;
        }
        logActionEvent("action_exit", action, cancel_requested_.load() ? "canceled" : "done");
        setStatus(cancel_requested_.load() ? Status::Idle : Status::Done);
        return;
    }

    if (action.prestart_before_userctrl && !action.command.empty()) {
        if (action.stop_loco_before_start && control_backend_) {
            if (!control_backend_->stop_move(error)) {
                terminateRunningProcess(false);
                request_switch_back_after_pre_userctrl_failure();
                setFailed("StopMove before prestart handoff failed: " + error);
                return;
            }
            loco_stopped = true;
        }
        if (!clear_handoff_files()) {
            request_switch_back_after_pre_userctrl_failure();
            setFailed("failed to clear handoff files: " + error);
            return;
        }

        const std::string command = expandCommandTemplate(action);
        std::cout << "[MotionActionController] prestarting action before UserCtrl: "
                  << command << std::endl;
        if (!startCommandStringAsync(command, error)) {
            request_switch_back_after_pre_userctrl_failure();
            setFailed("prestart failed: " + error);
            return;
        }
        payload_prestarted = true;
        event_log_.log("payload_spawned",
                       {event_string("action", action.name),
                        event_int("action_seq", static_cast<long long>(action.action_seq)),
                        event_int("pid", static_cast<long long>(running_pid_.load())),
                        event_string("phase", "prestart"),
                        event_string("payload_event_log", action.payload_event_log)});

        if (!action.ready_file.empty()) {
            if (!waitForFile(action.ready_file, action.prestart_ready_timeout_sec, error)) {
                terminateRunningProcess(false);
                if (cancel_requested_.load()) {
                    logActionEvent("action_exit", action, "canceled_before_handoff");
                    setStatus(Status::Idle);
                    return;
                }
                request_switch_back_after_pre_userctrl_failure();
                setFailed("prestart ready timeout: " + error);
                return;
            }
            logActionEvent("ready_seen", action, "prestart");
            std::cout << "[MotionActionController] prestarted payload ready, switching to UserCtrl"
                      << std::endl;
        }
    }

    if (action.stop_loco_before_start && !loco_stopped && control_backend_) {
        if (!control_backend_->stop_move(error)) {
            if (payload_prestarted) {
                terminateRunningProcess(false);
            }
            request_switch_back_after_pre_userctrl_failure();
            setFailed("StopMove before UserCtrl handoff failed: " + error);
            return;
        }
    }

    event_log_.log("switch_userctrl_req",
                   {event_string("action", action.name),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_bool("publish_prearm", !payload_prestarted)});
    const bool switch_ok = switchToUserCtrl(error, !payload_prestarted);
    event_log_.log("switch_userctrl_ret",
                   {event_string("action", action.name),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_bool("ok", switch_ok),
                    event_string("reason", switch_ok ? "ok" : error)});
    if (!switch_ok) {
        if (payload_prestarted) {
            terminateRunningProcess(false);
        }
        if (cancel_requested_.load()) {
            logActionEvent("action_exit", action, "canceled_during_handoff");
            setStatus(Status::Idle);
            return;
        }
        request_switch_back_after_pre_userctrl_failure();
        setFailed(error);
        return;
    }
    if (payload_prestarted && !action.start_file.empty()) {
        if (!touchFile(action.start_file, error)) {
            terminateRunningProcess(false);
            std::string switch_error;
            if (!switchBackToOfficialLogged(switch_error, action.switch_back_mode,
                                            action, "start_file_failure")) {
                error += "; official-control fallback failed: " + switch_error;
            }
            setFailed("failed to signal prestarted payload: " + error);
            return;
        }
        logActionEvent("start_file_touched", action, "activate_prestarted_payload");
    }

    setStatus(Status::Running);
    const bool payload_ok = payload_prestarted
        ? waitRunningCommand(expandCommandTemplate(action), error, &action)
        : runActionPayload(action, error);
    if (!payload_ok) {
        const bool suppress_switch_back = suppress_switch_back_on_cancel_.load();
        const bool canceled = cancel_requested_.load();
        if (!suppress_switch_back) {
            std::string switch_error;
            switchBackToOfficialLogged(switch_error, action.switch_back_mode,
                                       action, "payload_failed");
        }
        if (canceled) {
            logActionEvent("action_exit", action, "canceled");
            setStatus(Status::Idle);
            return;
        }
        logActionEvent("action_exit", action, error);
        setFailed(error);
        return;
    }
    logActionEvent("action_exit", action, cancel_requested_.load() ? "canceled" : "done");

    if (action.switch_back && !suppress_switch_back_on_cancel_.load() &&
        !handoff_release_sent_.load()) {
        setStatus(Status::SwitchingBack);
        if (!switchBackToOfficialLogged(error, action.switch_back_mode,
                                        action, "action_complete")) {
            setFailed(error);
            return;
        }
    }

    setStatus(cancel_requested_.load() ? Status::Idle : Status::Done);
}

bool MotionActionController::switchToUserCtrl(std::string& error_message, bool publish_prearm)
{
    std::lock_guard<std::mutex> mode_lock(control_mode_mutex_);
    if (cancel_requested_.load()) {
        error_message = "motion action canceled before UserCtrl handoff";
        return false;
    }
    if (!control_backend_) {
        error_message = "control backend is null";
        return false;
    }
    return control_backend_->switch_to_user_ctrl(publish_prearm, error_message);
}

bool MotionActionController::switchBackToOfficial(std::string& error_message,
                                                  const std::string& mode)
{
    std::lock_guard<std::mutex> mode_lock(control_mode_mutex_);
    if (!control_backend_) {
        error_message = "control backend is null";
        return false;
    }
    return control_backend_->switch_to_internal(
        static_cast<action_go::runtime::OfficialControlMode>(internalModeId(mode)),
        error_message);
}

bool MotionActionController::switchBackToOfficialLogged(std::string& error_message,
                                                        const std::string& mode,
                                                        const ActionConfig& action,
                                                        const std::string& reason)
{
    event_log_.log("switch_back_req",
                   {event_string("action", action.name),
                    event_string("mode", normalizeInternalMode(mode)),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_string("reason", reason)});
    const bool ok = switchBackToOfficial(error_message, mode);
    event_log_.log("switch_back_ret",
                   {event_string("action", action.name),
                    event_string("mode", normalizeInternalMode(mode)),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_bool("ok", ok),
                    event_string("reason", ok ? reason : error_message)});
    if (ok) {
        if (reason.find("failure") != std::string::npos ||
            reason.find("failed") != std::string::npos ||
            reason.find("recovery") != std::string::npos) {
            event_log_.log("fallback_completed",
                           {event_string("action", action.name),
                            event_int("action_seq", static_cast<long long>(action.action_seq)),
                            event_string("mode", normalizeInternalMode(mode)),
                            event_string("reason", reason)});
        }
    }
    return ok;
}

bool MotionActionController::completeSwitchBack(const ActionConfig& action,
                                                const std::string& mode,
                                                const std::string& reason,
                                                std::string& error_message)
{
    std::lock_guard<std::mutex> handoff_lock(switch_back_mutex_);
    if (handoff_release_sent_.load()) return true;
    setStatus(Status::SwitchingBack);
    event_log_.log("switch_back_req",
                   {event_string("action", action.name),
                    event_string("mode", normalizeInternalMode(mode)),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_string("reason", reason)});
    const bool ok = switchBackToOfficial(error_message, mode);
    event_log_.log("switch_back_ret",
                   {event_string("action", action.name),
                    event_string("mode", normalizeInternalMode(mode)),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_bool("ok", ok),
                    event_string("reason", ok ? reason : error_message)});
    if (!ok) {
        event_log_.log("switch_back_blocked",
                       {event_string("action", action.name),
                        event_int("action_seq", static_cast<long long>(action.action_seq)),
                        event_string("mode", normalizeInternalMode(mode)),
                        event_string("reason", error_message)});
        setStatus(Status::SwitchBackBlocked);
        return false;
    }

    event_log_.log("handoff_overlap_started",
                   {event_string("action", action.name),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_number("overlap_sec", action.switch_back_overlap_sec),
                    event_string("reason", reason)});
    if (supervisor_platform_ && action.switch_back_overlap_sec > 0.0) {
        supervisor_platform_->sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(action.switch_back_overlap_sec)));
    }
    if (!action.handoff_release_file.empty()) {
        if (!touchFile(action.handoff_release_file, error_message)) {
            setStatus(Status::SwitchBackBlocked);
            event_log_.log("switch_back_blocked",
                           {event_string("action", action.name),
                            event_int("action_seq", static_cast<long long>(action.action_seq)),
                            event_string("reason", error_message)});
            return false;
        }
        logActionEvent("handoff_release_sent", action, reason);
    } else {
        logActionEvent("handoff_release_sent", action, reason + ":terminate_payload");
    }
    handoff_release_sent_.store(true);
    return true;
}

bool MotionActionController::enterOfficialDamping(std::string& error_message)
{
    std::lock_guard<std::mutex> mode_lock(control_mode_mutex_);
    if (!control_backend_) {
        error_message = "control backend is null";
        return false;
    }
    return control_backend_->enter_damping(error_message);
}

bool MotionActionController::runActionPayload(const ActionConfig& action, std::string& error_message)
{
    if (cancel_requested_.load()) {
        error_message = "motion action canceled before start";
        return false;
    }

    if (!action.command.empty()) {
        return runExternalCommand(action, error_message);
    }

    // Temporary in-process placeholder: the switching/API/BT integration is
    // complete, and the BeyondMimic low-level runner can be plugged in here.
    // Configure duration_sec to test the full official->low-level->official
    // handoff without ROS.
    const double duration = action.duration_sec > 0.0 ? action.duration_sec : 1.0;
    std::cout << "[MotionActionController] action '" << action.name
              << "' has no command; holding UserCtrl for " << duration
              << " sec. model=" << action.model
              << " trajectory=" << action.trajectory << std::endl;
    return waitDurationOrCancel(duration);
}

bool MotionActionController::runExternalCommand(const ActionConfig& action, std::string& error_message)
{
    const std::string command = expandCommandTemplate(action);
    return runCommandString(action, command, error_message, "payload");
}

bool MotionActionController::runCommandString(const std::string& command, std::string& error_message)
{
    std::cout << "[MotionActionController] running action command: " << command << std::endl;
    if (!startCommandStringAsync(command, error_message)) {
        return false;
    }
    return waitRunningCommand(command, error_message);
}

bool MotionActionController::runCommandString(const ActionConfig& action,
                                              const std::string& command,
                                              std::string& error_message,
                                              const std::string& phase)
{
    std::cout << "[MotionActionController] running action command: " << command << std::endl;
    if (!startCommandStringAsync(command, error_message)) {
        return false;
    }
    event_log_.log("payload_spawned",
                   {event_string("action", action.name),
                    event_int("action_seq", static_cast<long long>(action.action_seq)),
                    event_int("pid", static_cast<long long>(running_pid_.load())),
                    event_string("phase", phase),
                    event_string("payload_event_log", action.payload_event_log)});
    return waitRunningCommand(command, error_message, &action);
}

bool MotionActionController::startCommandStringAsync(const std::string& command, std::string& error_message)
{
    std::string child_trial_id;
    std::string child_policy;
    action_go::runtime::ActionLifecycleEventV1 child_context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        child_trial_id = current_trial_id_;
        child_policy = current_action_;
        child_context = current_event_context_;
    }
    std::string child_command;
    auto export_context = [&](const char* key, const std::string& value) {
        if (!value.empty()) {
            child_command += "export ";
            child_command += key;
            child_command += "=";
            child_command += shellQuote(value);
            child_command += "; ";
        }
    };
    export_context("ACTION_GO_TRIAL_ID", child_trial_id);
    export_context("ACTION_GO_RUN_ID", child_context.run_id);
    export_context("ACTION_GO_PAIR_ID", child_context.pair_id);
    export_context("ACTION_GO_POLICY", child_context.policy.empty() ? child_policy : child_context.policy);
    export_context("ACTION_GO_HANDOFF_METHOD", child_context.handoff_method);
    export_context("ACTION_GO_GIT_SHA", child_context.git_sha);
    export_context("ACTION_GO_SPEC_SHA256", child_context.spec_sha256);
    export_context("ACTION_GO_MODEL_SHA256", child_context.model_sha256);
    export_context("ACTION_GO_HARDWARE_ID", child_context.hardware_id);
    export_context("ACTION_GO_MANIFEST_SHA256", child_context.manifest_sha256);
    child_command += command;
    if (!supervisor_platform_) {
        error_message = "supervisor platform is null";
        return false;
    }
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    pid_t pid = -1;
    if (!supervisor_platform_->start_shell(child_command, pid, error_message)) {
        return false;
    }
    running_pid_.store(pid);
    return true;
}

bool MotionActionController::waitRunningCommand(const std::string& command, std::string& error_message,
                                                const ActionConfig* action)
{
    const pid_t pid = running_pid_.load();
    if (pid <= 0) {
        error_message = "no running process to wait for";
        return false;
    }

    action_go::runtime::SupervisorProcessResult process_result;
    bool complete_seen = false;
    while (true) {
        if (action && action->postrun_handoff_hold && !complete_seen &&
            !action->handoff_complete_file.empty() &&
            supervisor_platform_->file_exists(action->handoff_complete_file)) {
            complete_seen = true;
            logActionEvent("handoff_hold_started", *action, "complete_file_seen");
            std::string handoff_error;
            if (dry_run_) {
                if (!touchFile(action->handoff_release_file, handoff_error)) {
                    error_message = handoff_error;
                    return false;
                }
                logActionEvent("handoff_release_sent", *action, "dry_run_no_control_switch");
                handoff_release_sent_.store(true);
            } else if (!completeSwitchBack(*action, action->switch_back_mode,
                                           "action_complete_hold", handoff_error)) {
                error_message = handoff_error;
                // Keep polling the payload in SwitchBackBlocked.  A later
                // external retry can complete the RPC and release the hold.
            }
        }
        {
            std::lock_guard<std::mutex> process_lock(process_mutex_);
            if (running_pid_.load() != pid) {
                if (cancel_requested_.load()) {
                    error_message = "motion action canceled";
                } else {
                    error_message = "action process is no longer available";
                }
                return false;
            }
            process_result = supervisor_platform_->poll_process(pid);
            if (process_result.state != action_go::runtime::SupervisorProcessState::kRunning) {
                running_pid_.store(-1);
            }
        }
        if (process_result.state == action_go::runtime::SupervisorProcessState::kExited) {
            break;
        }
        if (process_result.state == action_go::runtime::SupervisorProcessState::kError) {
            if (cancel_requested_.load()) {
                error_message = "motion action canceled";
            } else {
                error_message = process_result.error;
            }
            return false;
        }
        if (cancel_requested_.load()) {
            terminateRunningProcess(false);
            if (running_pid_.load() < 0) {
                continue;
            }
        }
        supervisor_platform_->sleep_for(std::chrono::milliseconds(20));
    }

    if (cancel_requested_.load()) {
        error_message = "motion action canceled";
        return false;
    }
    if (process_result.signal != 0 || process_result.exit_code != 0) {
        error_message = "action command failed exit_code=" +
                        std::to_string(process_result.exit_code) +
                        " signal=" + std::to_string(process_result.signal) + ": " + command;
        return false;
    }
    return true;
}

bool MotionActionController::waitForFile(const std::string& path, double timeout_sec, std::string& error_message)
{
    if (path.empty()) {
        return true;
    }
    const auto deadline = supervisor_platform_->now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(std::max(0.0, timeout_sec)));
    while (!cancel_requested_.load()) {
        const pid_t pid = running_pid_.load();
        if (pid > 0) {
            action_go::runtime::SupervisorProcessResult process_result;
            {
                std::lock_guard<std::mutex> process_lock(process_mutex_);
                if (running_pid_.load() != pid) continue;
                process_result = supervisor_platform_->poll_process(pid);
                if (process_result.state != action_go::runtime::SupervisorProcessState::kRunning) {
                    running_pid_.store(-1);
                }
            }
            if (process_result.state == action_go::runtime::SupervisorProcessState::kExited) {
                error_message = "process exited before creating file " + path +
                                " exit_code=" + std::to_string(process_result.exit_code) +
                                " signal=" + std::to_string(process_result.signal);
                return false;
            }
            if (process_result.state == action_go::runtime::SupervisorProcessState::kError) {
                error_message = "waitpid failed while waiting for file " + path +
                                ": " + process_result.error;
                return false;
            }
        }
        if (supervisor_platform_->file_exists(path)) {
            return true;
        }
        if (supervisor_platform_->now() >= deadline) {
            error_message = "timeout waiting for file: " + path;
            return false;
        }
        supervisor_platform_->sleep_for(std::chrono::milliseconds(10));
    }
    error_message = "canceled while waiting for file: " + path;
    return false;
}

bool MotionActionController::touchFile(const std::string& path, std::string& error_message)
{
    if (path.empty()) {
        return true;
    }
    return supervisor_platform_->touch_file(path, "start\n", error_message);
}

void MotionActionController::terminateRunningProcess(bool force)
{
    std::lock_guard<std::mutex> process_lock(process_mutex_);
    const pid_t pid = running_pid_.load();
    if (pid <= 0) {
        return;
    }
    std::string error;
    if (!supervisor_platform_->terminate_process_group(pid, force, error)) {
        std::cerr << "[MotionActionController] process termination warning: "
                  << error << std::endl;
    }
    running_pid_.store(-1);
}

void MotionActionController::joinWorkerIfJoinable()
{
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool MotionActionController::waitDurationOrCancel(double duration_sec)
{
    const auto deadline = supervisor_platform_->now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(duration_sec));
    while (!cancel_requested_.load() && supervisor_platform_->now() < deadline) {
        supervisor_platform_->sleep_for(std::chrono::milliseconds(10));
    }
    return !cancel_requested_.load();
}

void MotionActionController::setStatus(Status status)
{
    status_.store(status);
}

void MotionActionController::setFailed(const std::string& error_message)
{
    std::string action;
    uint64_t action_seq = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = error_message;
        action = current_action_;
        action_seq = current_action_seq_;
    }
    event_log_.log("action_failed",
                   {event_string("action", action),
                    event_int("action_seq", static_cast<long long>(action_seq)),
                    event_string("reason", error_message)});
    event_log_.log("fault_detected",
                   {event_string("action", action),
                    event_int("action_seq", static_cast<long long>(action_seq)),
                    event_string("component", "motion_action_controller"),
                    event_string("reason", error_message)});
    status_.store(Status::Failed);
    std::cerr << "[MotionActionController] FAILED: " << error_message << std::endl;
}

std::string MotionActionController::resolveResourcePath(const std::string& raw_path) const
{
    return resolveResourcePath(raw_path, {});
}

std::string MotionActionController::resolveResourcePath(const std::string& raw_path,
                                                        const std::string& base_dir) const
{
    if (raw_path.empty()) {
        return {};
    }

    fs::path path(raw_path);
    if (path.is_absolute()) {
        return path.lexically_normal().string();
    }

    if (!base_dir.empty()) {
        const fs::path from_base = (fs::path(base_dir) / path).lexically_normal();
        if (fs::exists(from_base)) {
            return from_base.string();
        }
    }

    const fs::path from_root = (action_go_root_ / path).lexically_normal();
    if (fs::exists(from_root)) {
        return from_root.string();
    }

    const fs::path from_config = (config_dir_ / path).lexically_normal();
    if (fs::exists(from_config)) {
        return from_config.string();
    }

    // Prefer action_go-root-relative paths even if the file does not exist yet,
    // so error messages and generated commands are stable before resources are copied.
    return from_root.string();
}

std::string MotionActionController::defaultPlayerPath() const
{
    const fs::path robocup_sdk = (action_go_root_ / ".." / "roboCup_sdk").lexically_normal();
    const fs::path in_tree_build = robocup_sdk / "build" / "action_go_build" / "generic_policy_player";
    if (fs::exists(in_tree_build)) {
        return in_tree_build.string();
    }

    const fs::path sibling_build = action_go_root_ / "build" / "generic_policy_player";
    if (fs::exists(sibling_build)) {
        return sibling_build.string();
    }

    return in_tree_build.string();
}

std::string MotionActionController::expandCommandTemplate(const ActionConfig& action) const
{
    return expandCommandTemplate(action, action.command);
}

std::string MotionActionController::expandCommandTemplate(const ActionConfig& action,
                                                          const std::string& command_template) const
{
    std::string command = command_template;
    replaceAll(command, "{name}", action.name);
    replaceAll(command, "{action_dir}", action.resource_base_dir);
    replaceAll(command, "{resource_base_dir}", action.resource_base_dir);
    replaceAll(command, "{action_go_root}", action_go_root_.string());
    replaceAll(command, "{config_dir}", config_dir_.string());
    replaceAll(command, "{player}", defaultPlayerPath());
    replaceAll(command, "{model}", action.model);
    replaceAll(command, "{params}", action.params);
    replaceAll(command, "{trajectory}", action.trajectory);
    replaceAll(command, "{net}", action.network_interface);
    replaceAll(command, "{network_interface}", action.network_interface);
    replaceAll(command, "{duration_sec}", std::to_string(action.duration_sec));
    replaceAll(command, "{max_frames}", std::to_string(action.max_frames));
    replaceAll(command, "{max_steps}", std::to_string(action.max_frames));
    replaceAll(command, "{motion_end_step}", std::to_string(action.max_frames));
    replaceAll(command, "{ready_file}", action.ready_file);
    replaceAll(command, "{start_file}", action.start_file);
    replaceAll(command, "{handoff_complete_file}", action.handoff_complete_file);
    replaceAll(command, "{handoff_release_file}", action.handoff_release_file);
    replaceAll(command, "{payload_event_log}", action.payload_event_log);
    replaceAll(command, "{supervisor_event_log}", action.supervisor_event_log);
    replaceAll(command, "{event_log_arg}", action.event_log_arg);
    return command;
}

std::string MotionActionController::dryRunCommandString(const ActionConfig& action) const
{
    std::string command = expandCommandTemplate(action);
    if (command.empty()) {
        return command;
    }

    // Most players implement --dry-run as "load model/DDS/lowstate but do not
    // publish LowCmd".  Instinct's useful no-LowCmd mode is --dds-dry-run,
    // because plain --dry-run uses zero depth and does not exercise RealSense.
    const bool already_dry =
        command.find("--dry-run") != std::string::npos ||
        command.find("--dds-dry-run") != std::string::npos ||
        command.find("--check-only") != std::string::npos ||
        command.find("--camera-check") != std::string::npos ||
        command.find("--depth-camera-check-only") != std::string::npos;
    if (!already_dry) {
        command += action.type == "instinct_parkour" ? " --dds-dry-run" : " --dry-run";
    }
    return command;
}
