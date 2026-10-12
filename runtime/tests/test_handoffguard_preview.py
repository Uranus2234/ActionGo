import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/sim"))
from handoffguard_preview import RiskSwitch


class Session:
    percent = 50.0

    def run(self, *_args):
        return [np.array([[self.percent]], dtype=np.float32)]


class RiskSwitchTest(unittest.TestCase):
    def setUp(self):
        self.gate = RiskSwitch.__new__(RiskSwitch)
        self.gate.session = Session()
        self.gate.history = np.zeros((1, 10, 93), dtype=np.float32)
        self.gate.next_log = float("inf")
        self.gate.reset()
        self.frame = np.zeros(93, dtype=np.float32)

    def tick(self, percent, now, stable=False):
        self.gate.session.percent = percent
        return self.gate.tick(self.frame, now, stable)

    def test_request_confirm_and_stable_reset(self):
        self.gate.request(0)
        self.assertIsNone(self.tick(34, 0))
        self.assertIsNone(self.tick(35, 0.02))
        self.assertIsNone(self.tick(34, 0.04))
        self.assertEqual(self.tick(34, 0.06), "bfm")
        for step in range(4):
            self.assertIsNone(self.tick(60, 0.08 + step * 0.02, True))
        self.assertIsNone(self.tick(60, 0.16, False))
        for step in range(4):
            self.assertIsNone(self.tick(60, 0.18 + step * 0.02, True))
        self.assertEqual(self.tick(60, 0.26, True), "walkrun")

    def test_deadline_keeps_source_and_allows_retry(self):
        self.gate.request(0)
        self.assertIsNone(self.tick(34, 1.98))
        self.assertIsNone(self.tick(34, 2.0))
        self.assertEqual(self.gate.state, "SOURCE")
        self.assertIsNone(self.tick(0, 2.02))
        self.gate.request(2.04)
        self.assertIsNone(self.tick(0, 2.04))
        self.assertEqual(self.tick(0, 2.06), "bfm")

    def test_accepts_risk_in_second_second(self):
        self.gate.request(0)
        self.assertIsNone(self.tick(80, 1.0))
        self.assertEqual(self.gate.state, "OBSERVE")
        self.assertIsNone(self.tick(34, 1.5))
        self.assertEqual(self.tick(34, 1.52), "bfm")


if __name__ == "__main__":
    unittest.main()
