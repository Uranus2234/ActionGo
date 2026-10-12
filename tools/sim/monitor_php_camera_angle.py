import argparse
from collections import deque
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time

import numpy as np
import yaml

from php_camera_geometry import estimate_lower_plane


def interrupt_monitor(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description="只读实时估算PHP相机俯仰角。")
    parser.add_argument("--net", default="eth0")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--duration-sec", type=float, default=86400)
    parser.add_argument("--latest-trace", type=Path)
    args = parser.parse_args()
    if args.latest_trace is not None:
        args.latest_trace.parent.mkdir(parents=True, exist_ok=True)
    active = subprocess.run([
        "pgrep", "-f", r"^(/home/unitree/action_go/)?(\./)?build(-php-dryrun)?/generic_sensor_policy_player( |$)"
    ], capture_output=True)
    if active.returncode == 0:
        raise RuntimeError("已有传感器策略运行，请先停止；角度监视不抢占相机。")
    signal.signal(signal.SIGTERM, interrupt_monitor)
    signal.signal(signal.SIGHUP, interrupt_monitor)
    print("PHP相机实时角度 | 只读DDS，无LowCmd发布、无控制权切换", flush=True)
    print("条件：脚下为水平地面，图像下部两侧可见地面。正角度表示向下俯。", flush=True)
    print("目标：相对躯干下俯27°；显示最近0.5秒中位数。运动测试前先Ctrl+C退出释放相机。", flush=True)
    with tempfile.TemporaryDirectory(prefix="actiongo-php-angle-") as directory:
        trace_path = Path(directory) / "inference.yaml"
        log_path = Path(directory) / "player.log"
        os.mkfifo(trace_path)
        descriptor = os.open(trace_path, os.O_RDWR | os.O_NONBLOCK)
        process = None
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen([
                    str(args.root / "build/generic_sensor_policy_player"),
                    "--spec", str(args.root / "actions/php/config/generic_php.yaml"),
                    "--net", args.net, "--dds-dry-run", "--duration-sec", str(args.duration_sec),
                    "--hz", "10", "--stats-interval", "10", "--command-source", "fixed",
                    "--command-index", "0", "--trace-file", str(trace_path),
                ], cwd=args.root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                buffered = b""
                angles = deque(maxlen=5)
                last_display = 0.0
                last_sample = time.monotonic()
                while True:
                    readable, _, _ = select.select([descriptor], [], [], 0.5)
                    if readable:
                        buffered += os.read(descriptor, 65536)
                        while b"\n" in buffered:
                            line, buffered = buffered.split(b"\n", 1)
                            record = yaml.load(line, Loader=yaml.CSafeLoader)
                            plane = estimate_lower_plane(record)
                            last_sample = time.monotonic()
                            if plane is None:
                                angles.clear()
                            else:
                                angles.append(plane["approx_camera_pitch_relative_to_torso_deg"])
                            if last_sample - last_display >= 0.5:
                                if args.latest_trace is not None:
                                    args.latest_trace.write_bytes(line + b"\n")
                                clock = time.strftime("%H:%M:%S")
                                if plane is None:
                                    print(f"[{clock}] 地面不足或被遮挡，当前无法估算；请保持水平地面可见。", flush=True)
                                else:
                                    angle = float(np.median(angles))
                                    difference = angle - 27
                                    status = "在训练角度范围内" if abs(difference) <= 3 else "超出训练角度范围"
                                    print(
                                        f"[{clock}] 相对躯干下俯≈{angle:5.2f}° | 目标27.00° | 偏差{difference:+5.2f}° | "
                                        f"光轴/水平={plane['optical_pitch_below_horizontal_deg']:.2f}° | "
                                        f"躯干pitch={plane['torso_pitch_deg']:+.2f}° | {status}", flush=True
                                    )
                                last_display = last_sample
                    if process.poll() is not None:
                        if process.returncode:
                            raise RuntimeError(log_path.read_text()[-3000:])
                        break
                    if time.monotonic() - last_sample > 1:
                        angles.clear()
                        print("等待新鲜深度/IMU数据，暂停角度显示……", flush=True)
                        last_sample = time.monotonic()
        except KeyboardInterrupt:
            print("\n停止角度监视，释放深度相机。", flush=True)
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            os.close(descriptor)


if __name__ == "__main__":
    main()
