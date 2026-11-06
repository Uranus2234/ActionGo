import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/sim"))
from stand_bridge import SimStandBridge


class StandBridgeTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        clock = patch.object(time, "monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.ready = False
        self.preparations = []
        self.commits = []
        self.manager = SimpleNamespace(
            actions={"bfm_zero_stand": object(), "target": object()},
            active_action="source", active_proc=Mock(poll=Mock(return_value=None)),
            pending_action=None, pending_walkrun=None, passive=False,
            notice=Mock(), validate_handoff=Mock(), discard_handoff=Mock(), enter_passive=Mock(),
            stop_active=Mock(), start_walkrun=Mock(), walkrun_running=Mock(return_value=False),
            handoff_ready=lambda prepared: self.ready, action_running=lambda: True,
        )
        self.manager.prepare_handoff = self.prepare
        self.manager.commit_handoff = self.commit
        self.bridge = SimStandBridge(self.manager)

    def prepare(self, name):
        self.preparations.append(name)
        return {"name": name, "proc": Mock(poll=Mock(return_value=None))}

    def commit(self, prepared):
        self.commits.append(prepared["name"])
        self.manager.active_action = prepared["name"]

    def tick(self, tilt=0.0, gyro=0.0, valid=True, stale=False):
        sample = SimpleNamespace(valid=valid, tilt_deg=tilt, stamp=self.now - (0.2 if stale else 0.0))
        return self.bridge.tick(sample, gyro)

    def activate_bfm(self, target="walkrun"):
        self.assertTrue(self.bridge.request(target))
        self.ready = True
        self.tick()
        self.assertEqual(self.commits, ["bfm_zero_stand"])

    def test_preparation_never_replaces_unready_source(self):
        self.assertTrue(self.bridge.request("target"))
        self.tick()
        self.assertEqual(self.manager.active_action, "source")
        self.assertEqual(self.commits, [])
        self.assertFalse(self.bridge.request("walkrun"))
        self.assertFalse(self.bridge.request("unknown"))

    def test_unsupported_target_is_rejected_before_bfm_prestart(self):
        self.manager.validate_handoff.side_effect = ValueError("unsupported target")
        self.assertFalse(self.bridge.request("target"))
        self.assertEqual(self.preparations, [])
        self.assertEqual(self.manager.active_action, "source")
        self.assertEqual(self.bridge.target, "")

    def test_five_samples_and_walkrun_ack_before_bfm_stop(self):
        self.activate_bfm()
        for index in range(4):
            self.now += 0.02
            self.assertIsNone(self.tick())
        self.tick()
        self.manager.start_walkrun.assert_not_called()
        self.now += 0.02
        self.tick()
        self.manager.start_walkrun.assert_called_once_with(allow_action=True)
        self.manager.stop_active.assert_not_called()
        self.tick()
        self.manager.stop_active.assert_not_called()
        self.manager.walkrun_running.return_value = True
        self.assertEqual(self.tick(), "walkrun")
        self.manager.stop_active.assert_called_once_with("none")
        self.assertEqual(self.bridge.target, "")

    def test_stand_resets_on_bad_or_stale_input(self):
        self.activate_bfm()
        for fields in ({"tilt": float("nan")}, {"gyro": float("nan")},
                       {"gyro": 0.3}, {"tilt": 15.0}, {"stale": True}, {"valid": False}):
            for index in range(4):
                self.now += 0.02
                self.tick()
            self.tick(**fields)
            self.assertEqual(self.bridge.steps, 0)
        self.manager.start_walkrun.assert_not_called()

    def test_target_requires_both_standing_and_fresh_commands(self):
        self.activate_bfm("target")
        self.assertEqual(self.preparations, ["bfm_zero_stand", "target"])
        self.ready = False
        for index in range(6):
            self.now += 0.02
            self.tick()
        self.assertEqual(self.commits, ["bfm_zero_stand"])
        self.ready = True
        self.assertEqual(self.tick(), "target")
        self.assertEqual(self.commits, ["bfm_zero_stand", "target"])
        self.manager.stop_active.assert_not_called()

    def test_timeout_preserves_source_before_bfm_but_damps_after(self):
        self.bridge.request("walkrun")
        self.now += 16.0
        self.tick()
        self.manager.enter_passive.assert_not_called()
        self.assertEqual(self.manager.active_action, "source")
        self.activate_bfm()
        self.now += 16.0
        self.tick()
        self.manager.enter_passive.assert_called_once()

    def test_cancel_removes_prepared_target_and_cannot_resume(self):
        self.activate_bfm("target")
        self.bridge.cancel()
        for index in range(8):
            self.now += 0.02
            self.tick()
        self.assertEqual(self.commits, ["bfm_zero_stand"])
        self.manager.discard_handoff.assert_called_once()

    def test_source_exit_cancels_preparation(self):
        self.bridge.request("walkrun")
        self.ready = True
        self.manager.active_proc.poll.return_value = 1
        self.tick()
        self.assertEqual(self.commits, [])
        self.assertEqual(self.bridge.target, "")


if __name__ == "__main__":
    unittest.main()
