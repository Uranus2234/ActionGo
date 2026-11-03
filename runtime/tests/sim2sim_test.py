import ast
import contextlib
import io
import json
import os
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/sim"))
from sim2sim_support import MOTOR_NAMES, SimSession, motor_layout, resource_overlay, signal_exists, walkrun_pose_accepted, write_signal


def command_adapter():
    source = ROOT / "tools/sim/action_go_sim_remote.py"
    module = ast.parse(source.read_text())
    selected = [node for node in module.body if
                isinstance(node, ast.FunctionDef) and node.name in {
                    "_set_cli_option", "_shell_join", "adapt_command_for_sim"} or
                isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id == "SIM_PUBLISH_HZ_CAPABLE_ACTION_TYPES" for target in node.targets)]
    namespace = {"shlex": shlex, "Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["adapt_command_for_sim"]


class Sim2SimTest(unittest.TestCase):
    def test_display_reports_command_fallback_as_damping(self):
        source = ROOT / "tools/sim/action_go_sim_remote.py"
        module = ast.parse(source.read_text())
        manager = next(node for node in module.body if isinstance(node, ast.ClassDef) and
                       node.name == "SimActionManager")
        display = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and
                       node.name == "control_display")
        namespace = {}
        exec(compile(ast.Module(body=[display], type_ignores=[]), str(source), "exec"), namespace)
        controller = SimpleNamespace(passive=False, bridge=SimpleNamespace(command_fallback=True))
        self.assertEqual(namespace["control_display"](controller), ("DAMPING", "", ""))

    def test_deferred_inputs_are_not_reported_as_ready(self):
        source = ROOT / "tools/sim/action_go_sim_remote.py"
        module = ast.parse(source.read_text())
        deferred = next(node for node in module.body if isinstance(node, ast.FunctionDef) and
                        node.name == "sim_deferred_reason")
        namespace = {"Path": Path, "shlex": shlex, "SIM_DISABLED_ACTION_TYPES": {"instinct_parkour"}}
        exec(compile(ast.Module(body=[deferred], type_ignores=[]), str(source), "exec"), namespace)
        reason = namespace["sim_deferred_reason"]
        self.assertEqual(reason(SimpleNamespace(name="kick_right", type="generic_policy",
                                               command="generic_policy_player")), "")
        self.assertIn("depth/object", reason(SimpleNamespace(name="pbfm_loco", type="generic_policy",
                                                   command="generic_sensor_policy_player --spec pbfm.yaml")))
        self.assertIn("realtime", reason(SimpleNamespace(name="tracking", type="generic_motion_tracking",
                                                        command="player --require-realtime")))
        self.assertIn("excluded", reason(SimpleNamespace(name="uni_supermimic_tracking", type="generic_policy",
                                                        command="generic_policy_player")))

    def test_walkrun_acceptance_rejects_fallen_and_invalid_states(self):
        self.assertTrue(walkrun_pose_accepted(0.78, 3.0, 0.5, 30.0))
        self.assertFalse(walkrun_pose_accepted(0.09, 95.0, 0.5, 30.0))
        self.assertFalse(walkrun_pose_accepted(0.78, 60.0, 0.5, 30.0))
        self.assertFalse(walkrun_pose_accepted(float("nan"), 3.0, 0.5, 30.0))
        self.assertFalse(walkrun_pose_accepted(0.78, float("nan"), 0.5, 30.0))

    def test_actuator_order_is_resolved_by_joint_name(self):
        joints = ["root", *["robot/" + name for name in MOTOR_NAMES], "object_hinge"]
        actuators = [30, *range(29, 0, -1)]
        self.assertEqual(motor_layout(joints, actuators), list(range(29, 0, -1)))
        with self.assertRaisesRegex(RuntimeError, "left_hip_pitch_joint"):
            motor_layout(joints, actuators + [1])
        with self.assertRaisesRegex(RuntimeError, "left_hip_pitch_joint"):
            motor_layout(joints + ["other/left_hip_pitch_joint"], actuators)
        with self.assertRaises(RuntimeError):
            motor_layout(joints[:-2], actuators)

    def test_overlay_preserves_config_and_uses_external_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, assets, output = [root / name for name in ("source", "assets", "output")]
            source.mkdir()
            assets.mkdir()
            (source / "spec.yaml").write_text("version: first\n")
            (source / "pointer.onnx").write_text("version https://git-lfs.github.com/spec/v1\n")
            (assets / "pointer.onnx").write_bytes(b"verified weight")
            (assets / "missing.onnx").write_bytes(b"another weight")
            resource_overlay(source, assets, output)
            (source / "spec.yaml").write_text("version: second\n")
            self.assertEqual((output / "spec.yaml").read_text(), "version: first\n")
            self.assertEqual((output / "pointer.onnx").read_bytes(), b"verified weight")
            self.assertEqual((output / "missing.onnx").resolve(), assets / "missing.onnx")

    def test_private_signals_reject_links_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "start"
            target = Path(temporary) / "target"
            target.write_text("preserve")
            path.symlink_to(target)
            self.assertFalse(signal_exists(str(path)))
            with self.assertRaises(FileExistsError):
                write_signal(str(path), "start\n")
            self.assertEqual(target.read_text(), "preserve")
            path.unlink()
            write_signal(str(path), "start\n")
            self.assertTrue(signal_exists(str(path)))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_signal(str(path), "again\n")

    def test_adapter_keeps_native_topics_and_overrides_domain_and_provider(self):
        adapt = command_adapter()
        command = "/build/generic_policy_player --cmd-topic rt/user_lowcmd --domain-id=0 --publish-hz 100 --label 'two  spaces'"
        tokens = shlex.split(adapt(command, "generic_policy", 500, True, False, 42, "cpu"))
        self.assertEqual(tokens[tokens.index("--cmd-topic") + 1], "rt/user_lowcmd")
        self.assertIn("--domain-id=42", tokens)
        self.assertEqual(tokens[tokens.index("--publish-hz") + 1], "500")
        self.assertEqual(tokens[tokens.index("--ep") + 1], "cpu")
        self.assertEqual(tokens[tokens.index("--label") + 1], "two  spaces")

    def test_session_records_failures_and_displays_current_latency(self):
        console = io.StringIO()
        old_session = os.environ.get("ACTION_GO_SESSION_DIR")
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(console):
            with SimSession(Path(temporary), "normal") as session:
                session.metadata(xml="fixed.xml", domain_id=42)
                timing = session.directory / "timing.jsonl"
                Path(str(timing) + ".latest.json").write_text(json.dumps(
                    dict(t_steady_ns=time.monotonic_ns(), e2e_ms=3.2)))
                print("hidden diagnostic")
                session.render("WALKRUN", "unitree_rl_mjlab", str(timing))
                session.last_render = 0
                session.render("DAMPING", "", "")
                session.result = 1
            self.assertFalse(json.loads((session.directory / "summary.json").read_text())["clean_shutdown"])
            self.assertIn("hidden diagnostic", (session.directory / "runtime.output.log").read_text())
            self.assertTrue((Path(temporary) / "latest").is_symlink())
        self.assertNotIn("hidden diagnostic", console.getvalue())
        self.assertIn("E2E 3.2 ms", console.getvalue())
        self.assertIn("DAMPING", console.getvalue())
        self.assertEqual(os.environ.get("ACTION_GO_SESSION_DIR"), old_session)

    def test_session_exception_has_nonzero_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "startup failure"):
                with SimSession(Path(temporary), "quiet") as session:
                    raise RuntimeError("startup failure")
            summary = json.loads((session.directory / "summary.json").read_text())
            self.assertEqual(summary, {"clean_shutdown": False, "exit_code": 1})


if __name__ == "__main__":
    unittest.main()
