#include "action_go/runtime/bfm_action.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace action_go::runtime {

BfmDecodedAction decode_bfm_action(
    const std::vector<float>& raw_action, float rescale,
    const std::vector<float>& default_position, const std::vector<float>& action_scale,
    const std::vector<float>& lower_limits, const std::vector<float>& upper_limits) {
  const size_t count = default_position.size();
  if (raw_action.size() != count || action_scale.size() != count ||
      lower_limits.size() != count || upper_limits.size() != count ||
      !std::isfinite(rescale) || rescale <= 0.f) {
    throw std::runtime_error("BFM action dimensions or rescale are invalid");
  }
  BfmDecodedAction result;
  result.scaled_action.resize(count);
  result.target_position.resize(count);
  for (size_t joint = 0; joint < count; ++joint) {
    if (!std::isfinite(raw_action[joint])) throw std::runtime_error("BFM action contains NaN/Inf");
    result.scaled_action[joint] = rescale * std::clamp(raw_action[joint], -1.f, 1.f);
    result.target_position[joint] = std::clamp(
        default_position[joint] + result.scaled_action[joint] * action_scale[joint],
        lower_limits[joint], upper_limits[joint]);
  }
  return result;
}

}
