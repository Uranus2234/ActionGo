#pragma once

#include <cmath>
#include <stdexcept>

namespace action_go::runtime {

struct PrearmCommandValues {
  float position{0.0f};
  float velocity{0.0f};
  float kp{0.0f};
  float kd{1.5f};
  float torque{0.0f};
};

inline PrearmCommandValues validated_prearm_command_values() {
  const PrearmCommandValues values;
  for (float value : {values.position, values.velocity, values.kp, values.kd, values.torque}) {
    if (!std::isfinite(value)) throw std::runtime_error("prearm LowCmd contains NaN/Inf");
  }
  if (values.kp < 0.0f || values.kp > 100.0f || values.kd < 0.0f || values.kd > 20.0f) {
    throw std::runtime_error("prearm LowCmd gain is outside the shared runtime bound");
  }
  return values;
}

}  // namespace action_go::runtime
