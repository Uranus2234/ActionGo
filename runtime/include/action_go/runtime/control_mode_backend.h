#pragma once

#include <string>

namespace action_go::runtime {

// Official controller modes accepted by the Unitree G1 handoff API.
// The numeric values intentionally match Unitree's LAST/PASSIVE/WALKRUN
// protocol, but callers should use the typed values rather than raw integers.
enum class OfficialControlMode {
  kLast = 0,
  kPassive = 1,
  kWalkRun = 2,
};

// Hardware boundary used by MotionActionController.  Production code wraps
// Unitree's LocoClient; deterministic tests can provide a fake implementation
// without opening DDS channels or changing controller ownership.
class ControlModeBackend {
 public:
  virtual ~ControlModeBackend() = default;

  virtual const char* name() const = 0;
  virtual bool stop_move(std::string& error_message) = 0;
  virtual bool switch_to_user_ctrl(bool publish_prearm,
                                   std::string& error_message) = 0;
  virtual bool switch_to_internal(OfficialControlMode mode,
                                  std::string& error_message) = 0;
  virtual bool enter_damping(std::string& error_message) = 0;
};

const char* official_control_mode_name(OfficialControlMode mode);

}  // namespace action_go::runtime
