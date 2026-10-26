import importlib.util
from collections import deque
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("supermimic_mujoco", REPO / "tools/sim/supermimic_mujoco.py")
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)


def queue_bridge():
    bridge = SIM.DDSBridge.__new__(SIM.DDSBridge)
    bridge.condition = threading.Condition()
    bridge.pending = deque()
    bridge.latest_sequence = 0
    bridge.messages = 0
    bridge.duplicates = 0
    bridge.hold_messages = 0
    bridge.error = None
    return bridge


def command(sequence, active=True):
    return SimpleNamespace(
        reserve=[sequence, SIM.LOCKSTEP_MAGIC if active else 0],
        motor_cmd=[SimpleNamespace(q=0.3, dq=0.0, tau=0.0, kp=10.0, kd=1.0) for _ in range(29)],
    )


class SuperMimicDDSTest(unittest.TestCase):
    def test_repeated_commands_do_not_advance_policy(self):
        bridge = queue_bridge()
        for sequence in (1, 1, 1, 2, 1, 2, 3):
            bridge.receive(command(sequence))
        self.assertEqual([bridge.pop()["sequence"] for _ in range(3)], [1, 2, 3])
        self.assertIsNone(bridge.pop())
        self.assertEqual(bridge.duplicates, 4)

    def test_hold_is_excluded_and_sequence_gaps_fail(self):
        bridge = queue_bridge()
        bridge.receive(command(0, active=False))
        self.assertEqual(bridge.hold_messages, 1)
        self.assertIsNone(bridge.pop())
        bridge.receive(command(2))
        with self.assertRaisesRegex(RuntimeError, "sequence gap"):
            bridge.pop()

    def test_nonfinite_command_fails_before_physics(self):
        bridge = queue_bridge()
        message = command(1)
        message.motor_cmd[3].q = float("nan")
        bridge.receive(message)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            bridge.pop()

    def test_lowcmd_uses_implicit_pd_and_preserves_effort_limit(self):
        model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <option timestep="0.005" integrator="implicitfast" gravity="0 0 0"/>
          <worldbody><body><joint name="hinge"/><geom type="sphere" size="0.1"/></body></worldbody>
          <actuator><general joint="hinge" gainprm="10" biastype="affine"
            biasprm="0 -10 -1" forcelimited="true" forcerange="-5 5"/></actuator>
        </mujoco>''')
        data = mujoco.MjData(model)
        bridge = SIM.DDSBridge.__new__(SIM.DDSBridge)
        bridge.model, bridge.data = model, data
        bridge.indices = {"actuators": np.asarray([0])}
        bridge.effort = np.asarray([[-5.0, 5.0]])
        data.qpos[0], data.qvel[0] = 0.1, 0.3
        values = {"q": [0.3], "dq": [0.2], "tau": [0.4], "kp": [12.0], "kd": [2.0]}
        bridge.apply({key: np.asarray(value) for key, value in values.items()})
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.ctrl[0], 0.3)
        self.assertAlmostEqual(data.actuator_force[0], 2.6)
        self.assertAlmostEqual(model.actuator_biasprm[0, 2], -2.0)
        values["q"] = [100.0]
        bridge.apply({key: np.asarray(value) for key, value in values.items()})
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.actuator_force[0], 5.0)


if __name__ == "__main__":
    unittest.main()
