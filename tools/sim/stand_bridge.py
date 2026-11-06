import math
import time


class SimStandBridge:
    def __init__(self, manager):
        self.manager = manager
        self.target = ""
        self.prepared = None
        self.standing = False
        self.deadline = 0.0
        self.steps = 0
        self.last_stamp = 0.0
        self.returning = False
        self.source_proc = None

    def cancel(self):
        if self.prepared is not None:
            self.manager.discard_handoff(self.prepared)
        self.prepared = None
        self.target = ""
        self.standing = False
        self.steps = 0
        self.last_stamp = 0.0
        self.returning = False
        self.source_proc = None

    def request(self, target):
        if self.target or target == "bfm_zero_stand" or (target != "walkrun" and target not in self.manager.actions):
            return False
        if self.manager.pending_action is not None or self.manager.pending_walkrun is not None:
            return False
        try:
            if target != "walkrun":
                self.manager.validate_handoff(target)
            prepared = self.manager.prepare_handoff("bfm_zero_stand")
        except Exception as error:
            self.manager.notice("Stand bridge unavailable: " + str(error), "WARN")
            return False
        self.prepared = prepared
        self.target = target
        self.standing = False
        self.steps = 0
        self.last_stamp = 0.0
        self.deadline = time.monotonic() + 15.0
        self.source_proc = self.manager.active_proc
        self.manager.notice(f"Bridge: {self.manager.active_action or 'WALKRUN'} -> bfm_zero_stand -> {target}")
        return True

    def tick(self, imu, gyro_norm):
        if not self.target:
            return None
        now = time.monotonic()
        if not self.standing and self.source_proc is not None and self.source_proc.poll() is not None:
            self.cancel()
            self.manager.notice("Stand bridge canceled: source exited", "WARN")
            return None
        failed = (now >= self.deadline or
                  (self.prepared is not None and self.prepared["proc"].poll() is not None))
        if self.standing:
            failed = failed or self.manager.passive or not self.manager.action_running() or self.manager.active_action != "bfm_zero_stand"
        if failed:
            standing = self.standing
            self.cancel()
            self.manager.notice("Stand bridge failed or timed out", "ERROR")
            if standing:
                self.manager.enter_passive()
            return None
        if not self.standing:
            if not self.manager.handoff_ready(self.prepared):
                return None
            try:
                self.manager.commit_handoff(self.prepared)
                self.prepared = None
                self.standing = True
                if self.target != "walkrun":
                    self.prepared = self.manager.prepare_handoff(self.target)
            except Exception as error:
                self.cancel()
                self.manager.notice("Stand bridge activation failed: " + str(error), "ERROR")
                self.manager.enter_passive()
            return None
        if self.returning:
            if not self.manager.walkrun_running():
                return None
            self.target = ""
            self.manager.stop_active("none")
            self.cancel()
            self.manager.notice("Bridge complete -> walkrun")
            return "walkrun"
        valid = (imu.valid and math.isfinite(imu.tilt_deg) and math.isfinite(gyro_norm) and
                 0.0 <= math.radians(imu.tilt_deg) < 0.20 and 0.0 <= gyro_norm < 0.30 and
                 imu.stamp > 0.0 and 0.0 <= now - imu.stamp < 0.1)
        if not valid:
            self.steps = 0
            self.last_stamp = 0.0
            return None
        if imu.stamp > self.last_stamp:
            self.steps += 1
            self.last_stamp = imu.stamp
        if self.steps < 5 or (self.prepared is not None and not self.manager.handoff_ready(self.prepared)):
            return None
        target = self.target
        try:
            if target == "walkrun":
                self.manager.start_walkrun(allow_action=True)
                self.returning = True
                return None
            else:
                self.manager.commit_handoff(self.prepared)
                self.prepared = None
            self.cancel()
            self.manager.notice("Bridge complete -> " + target)
            return target
        except Exception as error:
            self.cancel()
            self.manager.notice("Bridge handoff failed: " + str(error), "ERROR")
            self.manager.enter_passive()
            return None
