import ast
import contextlib
import io
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]


def load_recovery_functions():
    source_path = REPO / "tools/sim/action_go_sim_remote.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run_simulation")
    names = {
        "reset_safety_state", "ensure_safety_recovery_running", "arm_shell",
        "start_action", "start_safety_recovery", "safety_tick",
    }
    functions = [node for node in ast.walk(main) if isinstance(node, ast.FunctionDef) and node.name in names]
    wrapper = ast.parse(
        "def create(remote_cfg, remote, manager):\n"
        "    shell_armed = False\n"
        "    lie_action_active = False\n"
        "    selected_safety_action = remote_cfg.safety.action\n"
        "    safety_state = SafetyState.NORMAL\n"
        "    fall_timer_active = recover_timer_active = False\n"
        "    fall_since = recover_since = time.monotonic()\n"
        "    last_safety_finish = time.monotonic() - remote_cfg.safety.cooldown_ms / 1000.0\n"
    ).body[0]
    wrapper.body.extend(functions)
    wrapper.body.extend(ast.parse(
        "def state():\n"
        "    return shell_armed, safety_state, recover_timer_active\n"
        "return arm_shell, safety_tick, start_action, state\n"
    ).body)
    safety_state = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SafetyState")
    compiled = ast.fix_missing_locations(ast.Module(body=[safety_state, wrapper], type_ignores=[]))
    namespace = {"time": time, "normalize_mode": lambda mode: mode}
    exec(compile(compiled, str(source_path), "exec"), namespace)
    return namespace["create"]


class Manager:
    def __init__(self):
        self.active_action = ""
        self.running = False
        self.start_success = True
        self.starts = []
        self.events = []
        self.actions = {"unsafe_motion": SimpleNamespace(safety_enable=False)}

    def action_running(self):
        return self.running

    def start_action(self, name, source):
        self.starts.append(name)
        self.events.append(("start", name))
        self.running = self.start_success
        self.active_action = name if self.running else ""
        return self.running

    def stop_active(self, switch_mode):
        self.running = False
        self.active_action = ""

    def enter_passive(self):
        self.stop_active("none")
        self.events.append(("mode", "passive"))

    def start_walkrun(self):
        self.events.append(("mode", "walkrun"))


class SimulationRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.clock = patch.object(time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.config = SimpleNamespace(
            safety=SimpleNamespace(
                enable=True, auto_recovery=False, action="bfm_zero_stand", return_mode="walkrun",
                fall_angle_deg=57.3, recover_angle_deg=30.0, fall_hold_ms=60,
                recover_hold_ms=1200, cooldown_ms=1000,
            ),
            lie_action="lie_down_beyondmimic",
        )
        self.sample = SimpleNamespace(valid=True, tilt_deg=0.0, stamp=self.now)
        self.remote = SimpleNamespace(imu=lambda: self.sample)
        self.manager = Manager()
        self.arm, self.tick, self.start, self.state = load_recovery_functions()(
            self.config, self.remote, self.manager
        )

    def observe(self, tilt, elapsed=0.0, valid=True, stale=False):
        self.now += elapsed
        self.sample.tilt_deg = tilt
        self.sample.valid = valid
        self.sample.stamp = self.now - (0.6 if stale else 0.0)
        self.tick()

    def test_arm_boundary_and_backend_selection(self):
        self.sample.tilt_deg = 29.9
        self.arm("Start", "bfm_zero_stand")
        self.assertEqual(self.manager.starts, [])
        self.sample.tilt_deg = 30.0
        self.arm("Left+B", "ufo_stand")
        self.assertEqual(self.manager.starts, ["ufo_stand"])
        self.arm("Start", "bfm_zero_stand")
        self.assertEqual(self.manager.starts, ["ufo_stand"])
        self.start("unsafe_motion", "remote")
        self.assertEqual(self.manager.starts, ["ufo_stand"])

    def test_manual_arm_preserves_feature_imu_rules(self):
        self.sample.valid = False
        self.sample.tilt_deg = 120.0
        self.arm("Start", "bfm_zero_stand")
        self.assertEqual(self.manager.starts, [])
        self.sample.valid = True
        self.sample.stamp = self.now - 1.0
        self.arm("Start", "bfm_zero_stand")
        self.assertEqual(self.manager.starts, ["bfm_zero_stand"])

    def test_default_fall_damps_then_manual_arm_recovers(self):
        self.arm("Start", "bfm_zero_stand")
        self.observe(57.3)
        self.observe(57.3, 0.059)
        self.assertEqual(self.manager.events, [])
        self.observe(57.3, 0.002)
        self.assertEqual(self.manager.events, [("mode", "passive")])
        self.assertEqual(self.state()[:2], (False, "damped"))
        self.arm("Left+B", "ufo_stand")
        self.assertEqual(self.manager.starts, ["ufo_stand"])

    def test_auto_recovery_damps_before_start(self):
        self.config.safety.auto_recovery = True
        self.arm("Start", "bfm_zero_stand")
        self.observe(60.0)
        self.observe(60.0, 0.061)
        self.assertEqual(self.manager.events, [("mode", "passive"), ("start", "bfm_zero_stand")])

    def test_fall_timer_resets_on_upright_but_preserves_telemetry_gap(self):
        self.arm("Start", "bfm_zero_stand")
        self.observe(60.0)
        self.observe(0.0, 0.05)
        self.observe(60.0, 0.02)
        self.assertEqual(self.manager.events, [])
        self.observe(60.0, 0.10, stale=True)
        self.assertEqual(self.manager.events, [])
        self.observe(60.0)
        self.assertEqual(self.manager.events, [("mode", "passive")])

    def test_action_safety_disable_is_respected(self):
        self.arm("Start", "bfm_zero_stand")
        self.manager.running = True
        self.manager.active_action = "unsafe_motion"
        self.observe(90.0)
        self.observe(90.0, 1.0)
        self.assertEqual(self.manager.events, [])

    def test_recovery_failure_damps_without_restart(self):
        self.sample.tilt_deg = 90.0
        self.arm("Start", "bfm_zero_stand")
        self.observe(100.0)
        self.assertEqual(self.state()[1], "recovering")
        self.manager.running = False
        self.observe(90.0)
        self.observe(90.0, 1.0)
        self.assertEqual(self.manager.starts, ["bfm_zero_stand"])
        self.assertEqual(self.state()[:2], (False, "damped"))
        self.assertEqual(self.manager.events[-1], ("mode", "passive"))

    def test_start_failure_damps_without_retry(self):
        self.manager.start_success = False
        self.sample.tilt_deg = 90.0
        self.arm("Start", "bfm_zero_stand")
        self.observe(90.0)
        self.assertEqual(self.state()[:2], (False, "damped"))
        self.assertEqual(self.manager.starts, ["bfm_zero_stand"])

    def test_stable_upright_handoff_and_cooldown(self):
        self.sample.tilt_deg = 90.0
        self.arm("Start", "bfm_zero_stand")
        self.observe(30.0)
        self.observe(30.0, 1.3, stale=True)
        self.assertFalse(self.state()[2])
        self.observe(30.0)
        self.observe(31.0, 1.3)
        self.assertFalse(self.state()[2])
        self.observe(30.0)
        self.observe(30.0, 1.19)
        self.assertNotIn(("mode", "walkrun"), self.manager.events)
        self.observe(30.0, 0.02)
        self.assertEqual(self.manager.events[-1], ("mode", "walkrun"))
        self.assertEqual(self.state()[:2], (True, "normal"))
        self.observe(60.0, 0.2)
        self.observe(60.0, 0.2)
        self.assertNotIn(("mode", "passive"), self.manager.events)
        self.observe(60.0, 0.7)
        self.observe(60.0, 0.061)
        self.assertEqual(self.manager.events[-1], ("mode", "passive"))


if __name__ == "__main__":
    unittest.main()
