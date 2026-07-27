#include "action_go/runtime/generic_policy_runtime.h"

int main(int argc, char** argv) {
  return action_go::runtime::run_generic_policy_runtime(
      argc, argv, action_go::runtime::GenericPolicyRuntimeScope::kMotionTracking);
}
