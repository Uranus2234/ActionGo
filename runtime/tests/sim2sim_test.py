import ast
import contextlib
import io
import json
import os
import shlex
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/sim"))
from sim2sim_support import MOTOR_NAMES, KeyboardRemoteInput, SimSession, keyboard_combo_name, keyboard_remote_input, motor_layout, resource_overlay, signal_exists, walkrun_pose_accepted, write_signal


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
    def test_keyboard_emulates_remote_buttons_chords_and_axes(self):
        mask, axes = keyboard_remote_input({ord("B"), ord("R"), ord("1"), 257, 265,
                                            ord("w"), ord("d"), ord("e")})
        expected = (1 << 0) | (1 << 9) | (1 << 2) | (1 << 12)
        self.assertEqual(mask, expected)
        self.assertEqual(axes, [1.0, 0.0, 1.0, 1.0])
        mask, axes = keyboard_remote_input({ord("l"), ord("1"), ord("t"), ord("a"), 340})
        self.assertEqual(mask, (1 << 1) | (1 << 3) | (1 << 8))
        self.assertEqual(axes, [0.0, 0.0, 0.0, 0.0])

    def test_held_movement_does_not_add_remote_buttons_to_chord(self):
        held = {ord("a")}
        self.assertEqual(keyboard_remote_input(held), (0, [-1.0, 0.0, 0.0, 0.0]))
        for key in "B1R":
            held.add(ord(key))
        mask, axes = keyboard_remote_input(held)
        self.assertEqual(mask, (1 << 0) | (1 << 9))
        self.assertEqual(axes, [-1.0, 0.0, 0.0, 0.0])
        mask, axes = keyboard_remote_input({ord("s"), ord("t")})
        self.assertEqual(mask, 1 << 3)
        self.assertEqual(axes, [0.0, 0.0, 0.0, -0.3])

    def test_remote_start_and_a_are_explicit_and_opposing_axes_cancel(self):
        mask, axes = keyboard_remote_input({ord("s"), ord("a"), 344, 257})
        self.assertEqual(mask, (1 << 2) | (1 << 8))
        self.assertEqual(axes, [0.0] * 4)
        self.assertEqual(keyboard_remote_input(set(map(ord, "wasdqe"))), (0, [0.0] * 4))
        self.assertEqual(keyboard_remote_input(set()), (0, [0.0] * 4))

    def test_keyboard_binding_names_match_conflict_free_input(self):
        self.assertEqual(keyboard_combo_name((1 << 0) | (1 << 8)), "R+1+Shift+A")
        self.assertEqual(keyboard_combo_name((1 << 2) | (1 << 3)), "Shift+S+T")
        self.assertEqual(keyboard_combo_name((1 << 4) | (1 << 14)), "R+2+Down")
        self.assertEqual(keyboard_combo_name((1 << 6) | (1 << 8)), "F+1+Shift+A")
        self.assertEqual(keyboard_combo_name((1 << 0) | (1 << 6)), "R+1+F+1")
        self.assertEqual(keyboard_combo_name((1 << 1) | (1 << 13)), "L+1+Right")

    def test_all_shoulder_buttons_use_matching_letter_and_digit(self):
        for letter, digit, button in (("r", "1", 0), ("l", "1", 1), ("r", "2", 4),
                                     ("l", "2", 5), ("f", "1", 6), ("f", "2", 7)):
            with self.subTest(letter=letter, digit=digit):
                self.assertEqual(keyboard_remote_input({ord(letter), ord(digit)}),
                                 (1 << button, [0.0] * 4))
        self.assertEqual(keyboard_remote_input(set(map(ord, "r1f")))[0], (1 << 0) | (1 << 6))
        self.assertEqual(keyboard_remote_input(set(map(ord, "f34")))[0], 0)

    def test_short_button_taps_survive_until_the_supervisor_consumes_them(self):
        keyboard = KeyboardRemoteInput()
        for repeat in range(2):
            keyboard.set_key(257, True)
            keyboard.set_key(257, False)
        for sample in range(10):
            self.assertEqual(keyboard.sample(), (0, [0.0] * 4))
        self.assertEqual([keyboard.consume_buttons() for sample in range(5)],
                         [1 << 2, 0, 1 << 2, 0, 0])

    def test_chord_transitions_survive_a_slow_frame_without_latching_axes(self):
        keyboard = KeyboardRemoteInput()
        keyboard.set_key(ord("w"), True)
        keyboard.set_key(ord("f"), True)
        keyboard.set_key(ord("1"), True)
        keyboard.set_key(340, True)
        keyboard.set_key(ord("a"), True)
        keyboard.set_key(ord("a"), False)
        keyboard.set_key(340, False)
        keyboard.set_key(ord("1"), False)
        keyboard.set_key(ord("f"), False)
        keyboard.set_key(ord("w"), False)
        self.assertEqual(keyboard.sample(), (0, [0.0] * 4))
        self.assertEqual([keyboard.consume_buttons() for sample in range(4)],
                         [1 << 6, (1 << 6) | (1 << 8), 1 << 6, 0])
        keyboard.set_key(257, True)
        keyboard.clear()
        self.assertEqual(keyboard.consume_buttons(), 0)
        self.assertEqual(keyboard.sample(), (0, [0.0] * 4))

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

    def test_failed_walkrun_reports_player_reason_and_enters_damping(self):
        source = ROOT / "tools/sim/action_go_sim_remote.py"
        module = ast.parse(source.read_text())
        manager = next(node for node in module.body if isinstance(node, ast.ClassDef) and
                       node.name == "SimActionManager")
        failure = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and
                       node.name == "poll_walkrun_failure")
        namespace = {"os": os, "json": json}
        exec(compile(ast.Module(body=[failure], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            event_log = Path(temporary) / "walkrun.jsonl"
            event_log.write_text("\n".join(json.dumps(record) for record in [
                {"event": "process_exit", "reason": "command safety rejected stale LowState age=0.201s"},
                {"event": "safety_stop", "reason": "exception_path_damping"},
            ]))
            controller = SimpleNamespace(walkrun_proc=Mock(), walkrun_event_log=str(event_log),
                                         enter_passive=Mock(), session=Mock(), notice=Mock())
            controller.walkrun_proc.poll.return_value = None
            self.assertFalse(namespace["poll_walkrun_failure"](controller))
            controller.enter_passive.assert_not_called()
            controller.walkrun_proc.poll.return_value = 1
            self.assertTrue(namespace["poll_walkrun_failure"](controller))
            controller.enter_passive.assert_called_once()
            self.assertIn("stale LowState", controller.notice.call_args.args[0])
            self.assertIn("Backspace to reset", controller.notice.call_args.args[0])
            controller.session.event.assert_called_once()

    def test_ready_handshake_keeps_lowstate_fresh_while_waiting(self):
        source = ROOT / "tools/sim/action_go_sim_remote.py"
        module = ast.parse(source.read_text())
        manager = next(node for node in module.body if isinstance(node, ast.ClassDef) and
                       node.name == "SimActionManager")
        wait_ready = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and
                          node.name == "_wait_ready")
        namespace = {"time": time, "STOP": threading.Event(), "ActionConfig": object,
                     "subprocess": subprocess,
                     "signal_exists": lambda path: Path(path).exists()}
        exec(compile(ast.Module(body=[wait_ready], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            ready_path = Path(temporary) / "ready"
            player = Mock()
            player.poll.side_effect = [None, None, None]
            bridge = SimpleNamespace(publish_lowstate=Mock(side_effect=lambda: ready_path.touch(exist_ok=True)))
            controller = SimpleNamespace(bridge=bridge)
            action = SimpleNamespace(prestart_before_userctrl=True, ready_file=str(ready_path),
                                     prestart_ready_timeout_sec=1.0, name="WALKRUN")
            self.assertTrue(namespace["_wait_ready"](controller, action, player))
            self.assertGreaterEqual(bridge.publish_lowstate.call_count, 1)

    def test_interactive_transition_waits_for_ready_without_blocking_physics(self):
        source = ROOT / "tools/sim/action_go_sim_remote.py"
        module = ast.parse(source.read_text())
        manager = next(node for node in module.body if isinstance(node, ast.ClassDef) and
                       node.name == "SimActionManager")
        advance = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and
                       node.name == "advance_transitions")
        namespace = {"time": time, "signal_exists": lambda path: Path(path).exists(),
                     "write_signal": write_signal}
        exec(compile(ast.Module(body=[advance], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            ready_path = Path(temporary) / "ready"
            action = SimpleNamespace(prestart_before_userctrl=True, ready_file=str(ready_path))
            player = Mock()
            player.poll.return_value = None
            bridge = SimpleNamespace(publish_lowstate=Mock())
            controller = SimpleNamespace(
                _poll_retiring_processes=Mock(),
                pending_action={"proc": player, "action": action, "name": "test_policy",
                                "source": "test", "deadline": time.monotonic() + 1,
                                "started_at": time.monotonic()},
                pending_walkrun=None,
                active_action="test_policy",
                active_config=action,
                active_proc=None,
                session=Mock(),
                bridge=bridge,
                _activate_action=Mock(),
            )
            namespace["advance_transitions"](controller)
            self.assertIsNotNone(controller.pending_action)
            controller._activate_action.assert_not_called()
            bridge.publish_lowstate.assert_not_called()
            ready_path.touch()
            namespace["advance_transitions"](controller)
            self.assertIsNone(controller.pending_action)
            controller._activate_action.assert_called_once_with(
                "test_policy", "test", action, player)

    def test_deferred_inputs_are_not_reported_as_ready(self):
        source = ROOT / "tools/sim/action_go_sim_remote.py"
        module = ast.parse(source.read_text())
        deferred = next(node for node in module.body if isinstance(node, ast.FunctionDef) and
                        node.name == "sim_deferred_reason")
        check_actions = next(node for node in module.body if isinstance(node, ast.FunctionDef) and
                             node.name == "check_sim_actions")
        namespace = {"Path": Path, "shlex": shlex, "SIM_DISABLED_ACTION_TYPES": {"instinct_parkour"}}
        exec(compile(ast.Module(body=[deferred], type_ignores=[]), str(source), "exec"), namespace)
        reason = namespace["sim_deferred_reason"]
        self.assertEqual(reason(SimpleNamespace(name="kick_right", type="generic_policy",
                                               command="generic_policy_player")), "")
        self.assertEqual(reason(SimpleNamespace(name="pbfm_loco", type="generic_policy",
                                               command="generic_sensor_policy_player --spec pbfm.yaml")), "")
        self.assertIn("realtime", reason(SimpleNamespace(name="pbfm_pico_teleop", type="generic_policy",
                                                        command="generic_sensor_policy_player")))
        self.assertIn("sensor variant", reason(SimpleNamespace(name="unknown_sensor", type="generic_policy",
                                                              command="generic_sensor_policy_player")))
        self.assertIn("realtime", reason(SimpleNamespace(name="tracking", type="generic_motion_tracking",
                                                        command="player --require-realtime")))
        self.assertIn("excluded", reason(SimpleNamespace(name="uni_supermimic_tracking", type="generic_policy",
                                                        command="generic_policy_player")))
        check_namespace = {**namespace, "sim_deferred_reason": reason}
        exec(compile(ast.Module(body=[check_actions], type_ignores=[]), str(source), "exec"), check_namespace)
        session = SimpleNamespace(notice=Mock())
        args = SimpleNamespace(check=["steadytray_balance"], object_body="", session=session)
        action = SimpleNamespace(name="steadytray_balance", type="generic_policy",
                                 command="generic_sensor_policy_player")
        self.assertEqual(check_namespace["check_sim_actions"](args, {action.name: action}), 0)
        session.notice.assert_called_once_with(
            "SKIP steadytray_balance: requires --object-body and a calibrated policy frame in the shared XML")

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

    def test_slow_disk_does_not_block_control_telemetry(self):
        entered = threading.Event()
        release = threading.Event()
        produced = threading.Event()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            with SimSession(Path(temporary), "quiet") as session:
                original_write = session.write_event

                def slow_write(record):
                    entered.set()
                    release.wait(timeout=2)
                    original_write(record)

                session.write_event = slow_write
                session.event("first_state")
                self.assertTrue(entered.wait(timeout=1))

                def produce():
                    session.event("next_state")
                    session.render("WALKRUN", "unitree_rl_mjlab", "")
                    produced.set()

                producer = threading.Thread(target=produce)
                producer.start()
                try:
                    self.assertTrue(produced.wait(timeout=0.2), "disk logging stalled the control thread")
                finally:
                    release.set()
                    producer.join(timeout=2)
            records = [json.loads(line) for line in (session.directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual([record["event"] for record in records[:2]], ["first_state", "next_state"])


if __name__ == "__main__":
    unittest.main()
