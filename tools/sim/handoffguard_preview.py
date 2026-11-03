from __future__ import annotations

import select
import sys
import termios
import time
import tty

import numpy as np


class TerminalKeys:
    def __init__(self):
        self.settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
        return self

    def poll(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return ""

    def __exit__(self, *_args):
        if self.settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


class RiskSwitch:
    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.history = np.zeros((1, 10, 93), dtype=np.float32)
        self.ready = False
        self.state = "SOURCE"
        self.deadline = 0.0
        self.low_steps = 0
        self.stable_steps = 0
        self.next_log = 0.0
        self.percent = 100.0
        self.latency_ms = 0.0

    def reset(self):
        self.ready = False
        self.state = "SOURCE"
        self.low_steps = self.stable_steps = 0

    def request(self, now):
        print(f"[bridge] key=1 received sim_t={now:.3f} state={self.state}", flush=True)
        if self.state != "SOURCE":
            print(f"[bridge] request ignored: state={self.state}; "
                  "OBSERVE/BFM already switching; after WALKRUN press v to replay source", flush=True)
            return
        self.state = "OBSERVE"
        self.deadline = now + 2.0
        self.low_steps = 0
        self.minimum_risk = float("inf")
        self.best_low_steps = 0
        print("[bridge] request: observe <=2s; require risk <35% for 2 steps", flush=True)

    def tick(self, frame, now, stand_ready):
        if not self.ready:
            self.history[:] = frame
            self.ready = True
        else:
            self.history[:, :-1] = self.history[:, 1:].copy()
            self.history[0, -1] = frame
        started = time.perf_counter()
        self.percent = float(self.session.run(["risk_percent"], {"history": self.history})[0].item())
        self.latency_ms = (time.perf_counter() - started) * 1000.0
        command = None
        if self.state == "OBSERVE":
            if now >= self.deadline:
                self.state = "SOURCE"
                self.low_steps = 0
                print(f"[bridge] timeout: keep original policy; min_risk={self.minimum_risk:.2f}% "
                      f"best_consecutive_low_steps={self.best_low_steps}/2", flush=True)
            else:
                self.minimum_risk = min(self.minimum_risk, self.percent)
                self.low_steps = self.low_steps + 1 if 0 <= self.percent < 35 else 0
                self.best_low_steps = max(self.best_low_steps, self.low_steps)
                if self.low_steps >= 2:
                    self.state = "BFM"
                    self.stable_steps = 0
                    command = "bfm"
                    print("[bridge] approved -> BFM zero (original latent)", flush=True)
        elif self.state == "BFM":
            self.stable_steps = self.stable_steps + 1 if stand_ready else 0
            if self.stable_steps >= 5:
                self.state = "WALKRUN"
                command = "walkrun"
                print("[bridge] stable 5 steps -> WALKRUN", flush=True)
        wall_now = time.monotonic()
        if wall_now >= self.next_log:
            self.next_log = wall_now + 0.1
            print(f"[risk] sim_t={now:.3f} risk={self.percent:.2f}% inference_ms={self.latency_ms:.3f} "
                  f"state={self.state} low_steps={self.low_steps} stable_steps={self.stable_steps}", flush=True)
        return command
