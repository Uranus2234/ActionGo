import argparse
import base64
from collections import deque
import json
import os
from pathlib import Path
import select
import shlex
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


class RelayedDepthPublisher:
    def __init__(self, name):
        from multiprocessing import shared_memory
        import zmq

        self.shm = shared_memory.SharedMemory(name=name, create=True, size=58 * 87 * 4)
        self.array = np.ndarray((58, 87), dtype=np.float32, buffer=self.shm.buf)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("ipc:///tmp/" + name + ".heartbeat")
        self.sequence = 0

    def write(self, record):
        frame = np.frombuffer(base64.b64decode(record["depth_f32"]), dtype="<f4").reshape(58, 87)
        np.copyto(self.array, frame)
        self.sequence = record["sequence"]
        self.beat("OK", record["time_unix"])

    def beat(self, state, captured=None):
        self.socket.send_json({"v": 1, "seq": self.sequence, "t_capture": captured or time.time(), "state": state})

    def close(self):
        self.beat("FAILED")
        self.socket.close(linger=0)
        self.context.term()
        self.shm.close()
        self.shm.unlink()


def interrupt_stream(signum, frame):
    raise KeyboardInterrupt


def estimate_lower_plane(record):
    depth = (np.asarray(record["depth"]).reshape(58, 87) + 0.5) * 2.7 + 0.3
    rows, columns = np.indices(depth.shape)
    focal = 53 / np.tan(np.deg2rad(89.5 / 2))
    ray_x = ((columns + 0.5) * 98 / 87 - 0.5 + 4 - 53) / focal
    ray_y = (rows + 2 - 30) / focal
    valid = (rows >= 35) & ((columns < 22) | (columns >= 65)) & (depth > 0.35) & (depth < 2.9)
    features = np.stack([ray_x[valid], ray_y[valid], np.ones(valid.sum())], axis=1)
    if len(features) < 200:
        return None
    inverse_depth = 1 / depth[valid]
    generator = np.random.default_rng(0)
    initial_plane = np.linalg.lstsq(features, inverse_depth, rcond=None)[0]
    initial_error = np.abs(features @ initial_plane - inverse_depth)
    initial_distance = 1 / max(np.linalg.norm(initial_plane), 1e-12)
    clean_floor = (np.sqrt(np.mean(initial_error ** 2)) < 0.006
                   and initial_plane[1] > 0 and 1.05 <= initial_distance <= 1.65)
    best = initial_error < 0.008 if clean_floor else np.zeros(len(features), dtype=bool)
    for _ in range(0 if clean_floor else 128):
        selected = generator.choice(len(features), 3, replace=False)
        plane, _, rank, _ = np.linalg.lstsq(features[selected], inverse_depth[selected], rcond=None)
        distance = 1 / max(np.linalg.norm(plane), 1e-12)
        if rank < 3 or plane[1] <= 0 or not 1.05 <= distance <= 1.65:
            continue
        inliers = np.abs(features @ plane - inverse_depth) < 0.008
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 200:
        return None
    if np.ptp(rows[valid][best]) < 10 or np.ptp(columns[valid][best]) < 12:
        return None
    plane = np.linalg.lstsq(features[best], inverse_depth[best], rcond=None)[0]
    residual = np.sqrt(np.mean((features[best] @ plane - inverse_depth[best]) ** 2))
    distance = 1 / np.linalg.norm(plane)
    if residual > 0.01 or plane[1] <= 0 or not 1.05 <= distance <= 1.65:
        return None
    quaternion = np.asarray(record["quat"], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    torso_pitch = np.rad2deg(np.arcsin(np.clip(
        2 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1]), -1, 1,
    )))
    optical_pitch = np.rad2deg(np.arctan2(plane[2], plane[1]))
    return {"optical_pitch_below_horizontal_deg": float(optical_pitch),
            "torso_pitch_deg": float(torso_pitch),
            "approx_camera_pitch_relative_to_torso_deg": float(optical_pitch - torso_pitch),
            "optical_lateral_tilt_deg": float(np.rad2deg(np.arctan2(plane[0], np.linalg.norm(plane[1:])))),
            "plane_distance_m": float(distance), "floor_inlier_count": int(best.sum()),
            "floor_inlier_fraction": float(best.mean()), "inverse_depth_rmse_per_m": float(residual),
            "assumption": "level floor; standing camera height 1.05 to 1.65 m"}


def send_policy_depth(args):
    signal.signal(signal.SIGTERM, interrupt_stream)
    signal.signal(signal.SIGHUP, interrupt_stream)
    history = deque(maxlen=5)
    sequence = 0
    path = Path(args.policy_socket)
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    bound = False
    try:
        receiver.bind(str(path))
        bound = True
        receiver.settimeout(0.5)
        print(json.dumps({"type": "started", "pid": os.getpid()}), flush=True)
        while True:
            try:
                packet = receiver.recv(65536)
                receiver.setblocking(False)
                try:
                    while True:
                        packet = receiver.recv(65536)
                except BlockingIOError:
                    pass
                finally:
                    receiver.settimeout(0.5)
            except socket.timeout:
                history.clear()
                print(json.dumps({"type": "waiting", "message": "等待运行中的 PHP 策略深度；不启动相机或策略"}), flush=True)
                continue
            if len(packet) != 36 + 58 * 87 * 4 or packet[:8] != b"AGDEPTH1":
                continue
            timestamp, torso, *quaternion = struct.unpack_from("<QI4f", packet, 8)
            if time.time() - timestamp / 1e9 > 1:
                history.clear()
                continue
            depth = np.frombuffer(packet, dtype="<f4", offset=36)
            plane = estimate_lower_plane({"depth": depth, "quat": quaternion})
            if plane is None:
                history.clear()
            else:
                history.append(plane["approx_camera_pitch_relative_to_torso_deg"])
                plane["smoothed_relative_pitch_deg"] = float(np.median(history))
            sequence += 1
            print(json.dumps({"type": "depth", "sequence": sequence, "time_unix": timestamp / 1e9,
                              "depth_f32": base64.b64encode(packet[36:]).decode("ascii"),
                              "plane": plane, "imu_source": "torso" if torso else "pelvis+waist"}), flush=True)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        receiver.close()
        if bound:
            path.unlink(missing_ok=True)


def send_depth(args):
    import yaml
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    active = subprocess.run(["pgrep", "-f", r"(^|/)generic_sensor_policy_player( |$)"], capture_output=True)
    if active.returncode == 0:
        raise RuntimeError("已有传感器策略运行，请先停止；转发不抢占相机。")
    signal.signal(signal.SIGTERM, interrupt_stream)
    signal.signal(signal.SIGHUP, interrupt_stream)
    ChannelFactoryInitialize(0, args.net)
    remote_lock = threading.Lock()
    output_lock = threading.Lock()
    remote_state = {"raw": bytes(40), "received": 0.0}
    remote_stop = threading.Event()

    def receive_remote(message):
        with remote_lock:
            remote_state["raw"] = bytes(message.wireless_remote)
            remote_state["received"] = time.monotonic()

    def send_record(record):
        with output_lock:
            print(json.dumps(record), flush=True)

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(receive_remote, 1)

    def forward_remote():
        sequence = 0
        while not remote_stop.is_set():
            with remote_lock:
                raw, received = remote_state["raw"], remote_state["received"]
            if received:
                sequence += 1
                try:
                    send_record({"type": "remote", "sequence": sequence,
                                 "source_age_sec": time.monotonic() - received,
                                 "mask": int.from_bytes(raw[2:4], "little"),
                                 "axes": [struct.unpack_from("<f", raw, offset)[0]
                                          for offset in (4, 8, 12, 20)]})
                except BrokenPipeError:
                    return
            remote_stop.wait(0.02)

    remote_thread = threading.Thread(target=forward_remote, daemon=True)
    remote_thread.start()
    history = deque(maxlen=5)
    with tempfile.TemporaryDirectory(prefix="php-depth-live-") as directory:
        trace = Path(directory) / "depth.yaml"
        log_path = Path(directory) / "player.log"
        os.mkfifo(trace)
        descriptor = os.open(trace, os.O_RDWR | os.O_NONBLOCK)
        process = None
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen([
                    str(args.root / "build/generic_sensor_policy_player"),
                    "--spec", str(args.root / "actions/perception/php/config/generic_php.yaml"),
                    "--net", args.net, "--dds-dry-run", "--duration-sec", "86400",
                    "--hz", "10", "--stats-interval", "10", "--command-source", "fixed",
                    "--command-index", "0", "--trace-file", str(trace),
                ], cwd=args.root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                send_record({"type": "started", "pid": os.getpid(), "player_pid": process.pid})
                buffered = b""
                sequence = 0
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
                                history.clear()
                            else:
                                history.append(plane["approx_camera_pitch_relative_to_torso_deg"])
                                plane["smoothed_relative_pitch_deg"] = float(np.median(history))
                            sequence += 1
                            depth = np.asarray(record["depth"], dtype="<f4")
                            send_record({"type": "depth", "sequence": sequence, "time_unix": time.time(),
                                              "depth_f32": base64.b64encode(depth.tobytes()).decode("ascii"),
                                              "plane": plane, "imu_source": record["imu_source"]})
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text()[-3000:])
                    if time.monotonic() - last_sample > 1:
                        history.clear()
                        send_record({"type": "waiting", "message": "等待新鲜深度/IMU"})
                        last_sample = time.monotonic()
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        finally:
            remote_stop.set()
            remote_thread.join(timeout=1)
            subscriber.Close()
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            os.close(descriptor)


def depth_image(normalized):
    from PIL import Image, ImageDraw

    fraction = np.clip(normalized + 0.5, 0, 1)
    anchors = np.array([[255, 55, 35], [255, 220, 50], [30, 225, 135],
                        [25, 145, 255], [35, 35, 100]], dtype=float)
    colors = np.stack([np.interp(fraction, np.linspace(0, 1, len(anchors)), anchors[:, channel])
                       for channel in range(3)], axis=-1).astype(np.uint8)
    image = Image.fromarray(colors).resize((696, 464), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for lower, upper in ((0, 22), (65, 87)):
        draw.rectangle((lower * 8, 35 * 8, upper * 8 - 1, 463), outline="white", width=2)
    return image


def view_depth(args):
    import tkinter as tk
    from PIL import ImageTk

    args.output.mkdir(parents=True, exist_ok=True)
    ssh_options = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
    if args.control_path:
        ssh_options += ["-S", args.control_path]
    ssh_environment = os.environ.copy()
    ssh_environment.pop("LD_LIBRARY_PATH", None)
    ssh_environment.pop("LD_PRELOAD", None)
    command = shlex.join(["python3", "-u", str(args.remote_script), "--send", "--root", str(args.root), "--net", args.net])
    if args.policy:
        command += " --policy --policy-socket " + shlex.quote(args.policy_socket)
    process = subprocess.Popen(ssh_options + [args.host, command], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, bufsize=1, env=ssh_environment)
    lock = threading.Lock()
    shared = {"record": None, "received": 0.0, "remote": None, "remote_pid": None, "error": ""}

    def read_stream():
        publisher = None
        try:
            if args.shm_name:
                publisher = RelayedDepthPublisher(args.shm_name)
                print("真实深度写入共享内存：" + args.shm_name + "，维度58×87，保持原归一化。", flush=True)
            for line in process.stdout:
                record = json.loads(line)
                if record["type"] == "remote":
                    record["received_monotonic"] = time.monotonic()
                    temporary = args.output / "remote.tmp"
                    temporary.write_text(json.dumps(record))
                    temporary.replace(args.output / "remote.json")
                if publisher:
                    if record["type"] == "depth":
                        publisher.write(record)
                    elif record["type"] == "waiting":
                        publisher.beat("RECONNECTING")
                with lock:
                    if record["type"] == "started":
                        shared["remote_pid"] = record["pid"]
                    elif record["type"] == "depth":
                        shared["record"], shared["received"] = record, time.monotonic()
                    elif record["type"] == "remote":
                        shared["remote"] = record
            error = process.stderr.read().strip()
            print(error or "转发连接已结束", flush=True)
            with lock:
                shared["error"] = error or "转发连接已结束"
        except Exception as error:
            with lock:
                shared["error"] = str(error)
        finally:
            if publisher:
                publisher.close()

    threading.Thread(target=read_stream, daemon=True).start()
    window = tk.Tk()
    window.title("机器人 PHP 实时深度与相机角度")
    window.geometry("740x820+30+60")
    title = tk.Label(window, text="机器人 D435i → SSH 实时转发 | PHP 实际输入 58×87" +
                     ("\n读取运行中的策略；不占相机" if args.policy else "") +
                     ("\n同时供本机仿真策略读取" if args.shm_name else ""), font=("Sans", 14))
    title.pack(pady=8)
    image_label = tk.Label(window)
    image_label.pack()
    tk.Label(window, text="红色近 0.3m → 黄色 → 绿色 → 蓝色远 3.0m | 白框：地面拟合区域", font=("Sans", 11)).pack(pady=5)
    status = tk.Label(window, text="正在等待深度与躯干 IMU……", font=("Sans", 15), justify="left")
    status.pack(pady=8)
    remote_status = tk.Label(window, text="遥控器：等待 LowState", font=("Sans", 12))
    if not args.policy:
        remote_status.pack()
    monitor_note = ("旁路监控：可与 PHP 策略同时运行；关闭窗口不停止策略。运动/箱面会影响角度拟合。"
                    if args.policy else "只读推理，无电机命令；真机运动测试前关闭本窗口释放相机。")
    tk.Label(window, text="角度估算假设站在水平地面，白框需有足够地面。目标：相对躯干下俯 27°（±3°）。\n" + monitor_note,
             font=("Sans", 11), justify="left").pack(pady=5)
    closed = False
    last_sequence = 0
    last_display = 0.0
    rate_samples = deque(maxlen=20)

    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    signal.signal(signal.SIGTERM, lambda signum, frame: close())
    signal.signal(signal.SIGINT, lambda signum, frame: close())
    tk.Button(window, text="停止旁路转发" if args.policy else "停止转发并释放相机", command=close, font=("Sans", 12)).pack(pady=5)

    def update():
        nonlocal last_sequence, last_display
        if closed:
            return
        with lock:
            record, received, error = shared["record"], shared["received"], shared["error"]
            remote = shared["remote"]
        now = time.monotonic()
        if remote and now - remote["received_monotonic"] <= 0.2 and remote["source_age_sec"] <= 0.2:
            remote_status.configure(text="真机遥控：前后 %+.2f | 左右 %+.2f | R1 %s" % (
                remote["axes"][3], remote["axes"][0], bool(remote["mask"] & 1)), fg="#187534")
        else:
            remote_status.configure(text="遥控数据未连接或过期；仿真回中", fg="red")
        if error or (received and now - received > 1):
            status.configure(text=error or "数据超过1秒未更新，暂停角度显示", fg="red")
        elif record is not None and record["sequence"] != last_sequence:
            rate_samples.append((now, record["sequence"]))
            rate = ((rate_samples[-1][1] - rate_samples[0][1]) / (now - rate_samples[0][0])
                    if len(rate_samples) > 1 else 0)
            normalized = np.frombuffer(base64.b64decode(record["depth_f32"]), dtype="<f4").reshape(58, 87)
            image = depth_image(normalized)
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
            plane = record["plane"]
            header = "更新 %.1f Hz | 帧 %d | IMU %s" % (rate, record["sequence"], record["imu_source"])
            if plane is None:
                text = header + "\n地面不足、被遮挡或不平，暂不能可靠估算角度"
                color = "#a66b00"
            else:
                angle = plane["smoothed_relative_pitch_deg"]
                difference = angle - 27
                text = header + "\n相对躯干下俯 %.2f° | 偏差 %+.2f°\n光轴/水平 %.2f° | 躯干 pitch %+.2f°\n横向倾角 %+.2f° | 离地约 %.3f m\n地面点 %d | 拟合 RMSE %.4f /m" % (
                    angle, difference, plane["optical_pitch_below_horizontal_deg"],
                    plane["torso_pitch_deg"], plane["optical_lateral_tilt_deg"],
                    plane["plane_distance_m"], plane["floor_inlier_count"], plane["inverse_depth_rmse_per_m"])
                color = "#187534" if abs(difference) <= 3 else "#c02d22"
            status.configure(text=text, fg=color)
            if now - last_display >= 0.5:
                print(text.replace("\n", " | "), flush=True)
                (args.output / "latest.json").write_text(json.dumps({"sequence": record["sequence"],
                    "plane": plane, "receive_rate_hz": rate}, ensure_ascii=False, indent=2))
                image.save(args.output / "latest_depth.png")
                np.save(args.output / "latest_depth.npy", normalized)
                last_display = now
            last_sequence = record["sequence"]
        window.after(50, update)

    try:
        update()
        window.mainloop()
    finally:
        with lock:
            remote_pid = shared["remote_pid"]
        if remote_pid is not None and process.poll() is None:
            subprocess.run(ssh_options + [args.host, "kill -TERM " + str(int(remote_pid))], timeout=8,
                           check=False, env=ssh_environment)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print("已停止旁路转发；策略继续运行。" if args.policy else
              "已停止深度转发；机器人只读采集进程退出并释放相机。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="转发PHP实际深度并实时估算相机安装俯仰角。")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--policy", action="store_true", help="旁路读取运行中的 PHP 策略，不开启相机或策略")
    parser.add_argument("--policy-socket", default="/tmp/action_go_php_depth.sock")
    parser.add_argument("--root", type=Path, default=Path("/home/unitree/action_go"))
    parser.add_argument("--net", default="eth0")
    parser.add_argument("--host", default="unitree@192.168.3.162")
    parser.add_argument("--control-path")
    parser.add_argument("--shm-name", default="")
    parser.add_argument("--remote-script", type=Path, default=Path("/tmp/php_depth_live.py"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/php_depth_live")
    args = parser.parse_args()
    if args.send:
        send_policy_depth(args) if args.policy else send_depth(args)
    else:
        view_depth(args)


if __name__ == "__main__":
    main()
