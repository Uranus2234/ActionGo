#pragma once

namespace action_go::runtime {

enum class GenericPolicyRuntimeScope {
  kPolicy,
  kMotionTracking,
  kSensor,
};

int run_generic_policy_runtime(int argc, char** argv, GenericPolicyRuntimeScope scope);

}  // namespace action_go::runtime
