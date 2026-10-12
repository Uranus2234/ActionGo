#include "action_go/runtime/bridge_gate.h"

#include <cassert>

using action_go::runtime::BridgeCommand;
using action_go::runtime::BridgeGate;
using action_go::runtime::BridgeState;

int main() {
  BridgeGate gate;
  auto out = gate.tick(5.0F, true, false, false);
  assert(out.command == BridgeCommand::kNone);
  out = gate.tick(5.0F, true, false, false);
  assert(out.command == BridgeCommand::kStartBfm);
  assert(gate.state() == BridgeState::kBfmStand);
  for (int i = 0; i < 4; ++i) assert(gate.tick(5.0F, false, false, true).command == BridgeCommand::kNone);
  assert(gate.tick(5.0F, false, false, true).command == BridgeCommand::kSwitchToB);

  gate.reset();
  gate.tick(50.0F, false, true, false);
  for (int i = 0; i < 24; ++i) assert(gate.tick(50.0F, false, false, false).command == BridgeCommand::kNone);
  assert(gate.tick(50.0F, false, false, false).command == BridgeCommand::kDamping);
  return 0;
}
