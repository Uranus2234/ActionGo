#pragma once

#include <vector>

namespace action_go::runtime {

struct BfmDecodedAction {
  std::vector<float> scaled_action;
  std::vector<float> target_position;
};

BfmDecodedAction decode_bfm_action(const std::vector<float>& raw_action,
                                   float rescale,
                                   const std::vector<float>& default_position,
                                   const std::vector<float>& action_scale,
                                   const std::vector<float>& lower_limits,
                                   const std::vector<float>& upper_limits);

}
