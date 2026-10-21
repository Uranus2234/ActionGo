#pragma once

#include <vector>

namespace action_go::runtime {

std::vector<float> preprocess_php_depth(const std::vector<float>& depth_m);
std::vector<float> preprocess_pbfm_depth(const std::vector<float>& optical_z_m);

}
