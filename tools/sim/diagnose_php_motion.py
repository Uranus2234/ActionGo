import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_


def decode_remote(raw):
    buttons = int.from_bytes(bytes(raw[2:4]), "little")
    left = -struct.unpack_from("<f", bytes(raw), 4)[0]
    yaw = -struct.unpack_from("<f", bytes(raw), 8)[0]
    forward = struct.unpack_from("<f", bytes(raw), 20)[0]
    if math.hypot(forward, left) < 0.05:
        command = 0 if abs(yaw) < 0.05 else 3 if yaw > 0 else 5
    else:
        angle = math.degrees(math.atan2(left, forward))
        if -22.5 <= angle <= 22.5:
            command = 1
        elif 22.5 < angle <= 67.5:
            command = 2
        elif 67.5 < angle <= 112.5:
            command = 3
        elif -67.5 <= angle < -22.5:
            command = 4
        elif -112.5 <= angle < -67.5:
            command = 5
        else:
            command = 11
    if buttons & 1 and 1 <= command <= 5:
        command += 5
    return {"buttons": buttons, "R1": bool(buttons & 1), "forward": forward,
            "left": left, "yaw": yaw, "expected_php_command": command}


def main():
    parser = argparse.ArgumentParser(description="只读订阅遥控、LowState和LowCmd，不打开相机、不发布控制命令。")
    parser.add_argument("--net", default="eth0")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--sample-hz", type=float, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ChannelFactoryInitialize(0, args.net)
    lock = threading.Lock()
    latest, received, counts = {}, {}, Counter()
    interval_gaps, largest_gaps = {}, {}

    def callback(topic):
        def receive(message):
            now = time.monotonic()
            with lock:
                if topic in received:
                    gap_ms = (now - received[topic]) * 1000
                    interval_gaps[topic] = max(interval_gaps.get(topic, 0), gap_ms)
                    largest_gaps[topic] = max(largest_gaps.get(topic, 0), gap_ms)
                latest[topic] = message
                received[topic] = now
                counts[topic] += 1
        return receive

    subscribers = []
    for topic, message_type in (("rt/lowstate", LowState_), ("rt/user_lowcmd", LowCmd_),
                                ("rt/lowcmd", LowCmd_), ("rt/wirelesscontroller", WirelessController_),
                                ("rt/secondary_imu", IMUState_)):
        subscriber = ChannelSubscriber(topic, message_type)
        subscriber.Init(callback(topic), 1)
        subscribers.append(subscriber)
    print("只读监测：不接管控制、不发LowCmd、不占相机；可观察现有运行状态和手柄输入。", flush=True)
    started = previous = time.monotonic()
    previous_counts = Counter()
    command_counts = Counter()
    previous_positions = None
    next_print = started
    printed_at = started
    printed_counts = Counter()
    try:
        with args.output.open("w") as output:
            while time.monotonic() - started < args.duration:
                time.sleep(1 / args.sample_hz)
                now = time.monotonic()
                with lock:
                    messages, stamps, totals = latest.copy(), received.copy(), counts.copy()
                    gaps = interval_gaps.copy()
                    interval_gaps.clear()
                record = {"elapsed": now - started, "t_unix_ns": time.time_ns(),
                          "hz": {topic: (totals[topic] - previous_counts[topic]) / (now - previous)
                                 for topic in totals},
                          "age_ms": {topic: (now - stamp) * 1000 for topic, stamp in stamps.items()},
                          "max_receive_gap_ms": gaps}
                state = messages.get("rt/lowstate")
                if state is not None:
                    record["remote"] = decode_remote(state.wireless_remote)
                    command_counts[record["remote"]["expected_php_command"]] += 1
                    positions = [float(motor.q) for motor in state.motor_state[:29]]
                    velocities = [float(motor.dq) for motor in state.motor_state[:29]]
                    record["state"] = {"mode_machine": int(state.mode_machine), "leg_q": positions[:12],
                                       "leg_dq_max": max(map(abs, velocities[:12])),
                                       "q": positions, "dq": velocities,
                                       "quat": [float(value) for value in state.imu_state.quaternion],
                                       "gyro": [float(value) for value in state.imu_state.gyroscope],
                                       "motor_modes": [int(motor.mode) for motor in state.motor_state[:29]],
                                       "tau_est": [float(motor.tau_est) for motor in state.motor_state[:29]],
                                       "leg_tau_est": [float(motor.tau_est) for motor in state.motor_state[:12]]}
                    if previous_positions is not None:
                        record["state"]["leg_q_change"] = max(abs(current - old)
                                                              for current, old in zip(positions[:12], previous_positions[:12]))
                    previous_positions = positions
                    for topic in ("rt/user_lowcmd", "rt/lowcmd"):
                        command = messages.get(topic)
                        if command is None:
                            continue
                        motors = command.motor_cmd[:29]
                        desired_torque = [float(motor.tau + motor.kp * (motor.q - position)
                                                + motor.kd * (motor.dq - velocity))
                                          for motor, position, velocity in zip(motors, positions, velocities)]
                        record[topic] = {"mode_machine": int(command.mode_machine), "mode_pr": int(command.mode_pr),
                                         "motor_modes": [int(motor.mode) for motor in motors],
                                         "q": [float(motor.q) for motor in motors],
                                         "kp": [float(motor.kp) for motor in motors],
                                         "kd": [float(motor.kd) for motor in motors],
                                         "tau": [float(motor.tau) for motor in motors],
                                         "desired_torque": desired_torque,
                                         "desired_torque_max": max(map(abs, desired_torque))}
                torso = messages.get("rt/secondary_imu")
                if torso is not None:
                    record["torso_quat"] = [float(value) for value in torso.quaternion]
                    record["torso_gyro"] = [float(value) for value in torso.gyroscope]
                wireless = messages.get("rt/wirelesscontroller")
                if wireless is not None:
                    record["wireless_topic"] = {"buttons": int(wireless.keys), "lx": float(wireless.lx),
                                                 "ly": float(wireless.ly), "rx": float(wireless.rx)}
                output.write(json.dumps(record) + "\n")
                output.flush()
                remote = record.get("remote", {})
                if now >= next_print:
                    receive_hz = (totals["rt/user_lowcmd"] - printed_counts["rt/user_lowcmd"]) / (now - printed_at)
                    user_command = record.get("rt/user_lowcmd", {})
                    gains = user_command.get("kp", [])[:12]
                    torques = user_command.get("desired_torque", [])[:12]
                    print("t=%.1f command=%s R1=%s forward=%.3f usercmd_rx=%.0fHz age=%.0fms dq=%.3f leg_kp=[%.1f,%.1f] leg_torque=%.1f mode=%s" % (
                        record["elapsed"], remote.get("expected_php_command", "missing"), remote.get("R1", False),
                        remote.get("forward", 0), receive_hz,
                        record["age_ms"].get("rt/user_lowcmd", -1), record.get("state", {}).get("leg_dq_max", 0),
                        min(gains, default=0), max(gains, default=0), max(map(abs, torques), default=0),
                        user_command.get("mode_machine", "missing")), flush=True)
                    next_print = now + 1
                    printed_at, printed_counts = now, totals
                previous, previous_counts = now, totals
    except KeyboardInterrupt:
        pass
    finally:
        for subscriber in subscribers:
            subscriber.Close()
    print(json.dumps({"received": dict(counts), "largest_receive_gap_ms": largest_gaps,
                      "observed_remote_commands": dict(command_counts),
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
