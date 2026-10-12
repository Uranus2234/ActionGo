#!/usr/bin/env python3
"""
action_go 上位机 UDP 指令发送工具。

示例：
  python3 action_go_udp_send.py --host 192.168.123.255 --broadcast --action kick_right --delay-ms 500
  python3 action_go_udp_send.py --host 239.255.88.88 --action kick_right --delay-ms 500
  python3 action_go_udp_send.py --host 192.168.123.255 --broadcast --cmd normal_exit
"""

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import shlex
import socket
import sys
import time
from typing import List


def is_multicast(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_multicast
    except ValueError:
        return False


def looks_like_broadcast(host: str) -> bool:
    return host == "255.255.255.255" or host.endswith(".255")


def build_message(args: argparse.Namespace) -> str:
    if args.raw:
        return args.raw

    cmd = args.cmd
    if args.action:
        cmd = "action"

    if cmd == "action":
        if not args.action:
            raise SystemExit("ERROR: --cmd action requires --action ACTION_NAME")
        parts = ["ACTION", args.action]
        if args.start_after_ms is not None:
            start_ms = int(time.time() * 1000) + args.start_after_ms
            parts.append(f"start_ms={start_ms}")
        elif args.start_ms is not None:
            parts.append(f"start_ms={args.start_ms}")
        elif args.delay_ms is not None:
            parts.append(f"delay_ms={args.delay_ms}")
        return " ".join(parts)

    return cmd


def sign_message(message: str, args: argparse.Namespace) -> str:
    if not args.auth_key_file:
        return message
    with open(args.auth_key_file, "rb") as key_stream:
        key = key_stream.read().strip()
    if len(key) < 32:
        raise ValueError("--auth-key-file must contain at least 32 bytes")
    tokens = message.split()
    lowered = message.strip().lower()
    issued_at_ms = int(time.time() * 1000)
    execute_at_ms = issued_at_ms
    action = ""
    if lowered in ("emergency", "stop"):
        command_type = "emergency"
    elif lowered in ("damping", "damping_exit", "passive"):
        command_type = "damping_exit"
    elif lowered in ("normal_exit", "walkrun_exit", "walkrun", "exit"):
        command_type = "walkrun_exit"
    elif tokens and tokens[0].lower() in ("action", "run", "start") and len(tokens) >= 2:
        command_type = "action"
        action = tokens[1]
        for token in tokens[2:]:
            key_name, separator, value = token.partition("=")
            if not separator:
                raise ValueError(f"cannot sign unsupported command token: {token}")
            if key_name in ("delay_ms", "delay"):
                execute_at_ms = issued_at_ms + int(value)
            elif key_name in ("start_ms", "start_epoch_ms", "time_ms"):
                execute_at_ms = int(value)
            else:
                raise ValueError(f"cannot sign unsupported command field: {key_name}")
    else:
        raise ValueError("signed mode accepts registered actions and stop/damping/walkrun commands only")

    nonce = secrets.token_urlsafe(18)
    canonical = "\n".join([
        "1", command_type, action, str(issued_at_ms), str(execute_at_ms), nonce, args.auth_key_id,
    ])
    signature = hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return json.dumps({
        "version": 1,
        "type": command_type,
        "action": action,
        "issued_at_ms": issued_at_ms,
        "execute_at_ms": execute_at_ms,
        "nonce": nonce,
        "key_id": args.auth_key_id,
        "signature": signature,
    }, separators=(",", ":"), sort_keys=True)


def _parse_timing_tokens(tokens: List[str], default_delay_ms: int) -> List[str]:
    out: List[str] = []
    used_timing = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        key = t.lower().replace("-", "_")
        if key in ("now", "immediate"):
            used_timing = True
        elif key.startswith("delay_ms=") or key.startswith("delay="):
            out.append(f"delay_ms={t.split('=', 1)[1]}")
            used_timing = True
        elif key in ("--delay_ms", "delay_ms", "--delay", "delay"):
            if i + 1 >= len(tokens):
                raise ValueError(f"{t} requires a value")
            out.append(f"delay_ms={tokens[i + 1]}")
            used_timing = True
            i += 1
        elif key.startswith("start_ms=") or key.startswith("start_epoch_ms="):
            out.append(f"start_ms={t.split('=', 1)[1]}")
            used_timing = True
        elif key in ("--start_ms", "start_ms", "--start_epoch_ms", "start_epoch_ms"):
            if i + 1 >= len(tokens):
                raise ValueError(f"{t} requires a value")
            out.append(f"start_ms={tokens[i + 1]}")
            used_timing = True
            i += 1
        elif key.startswith("start_after_ms="):
            start_ms = int(time.time() * 1000) + int(t.split("=", 1)[1])
            out.append(f"start_ms={start_ms}")
            used_timing = True
        elif key in ("--start_after_ms", "start_after_ms"):
            if i + 1 >= len(tokens):
                raise ValueError(f"{t} requires a value")
            start_ms = int(time.time() * 1000) + int(tokens[i + 1])
            out.append(f"start_ms={start_ms}")
            used_timing = True
            i += 1
        else:
            raise ValueError(f"unknown option/token: {t}")
        i += 1

    if not used_timing and default_delay_ms >= 0:
        out.append(f"delay_ms={default_delay_ms}")
    return out


def build_interactive_message(line: str, default_delay_ms: int) -> str:
    tokens = shlex.split(line)
    if not tokens:
        return ""

    first = tokens[0].lower()
    if first in ("help", "?"):
        return "__HELP__"
    if first in ("q", "quit", "local_quit"):
        return "__QUIT__"
    if first in ("raw", "send"):
        if len(tokens) < 2:
            raise ValueError("raw requires a message, e.g. raw ACTION kick_right delay_ms=500")
        return " ".join(tokens[1:])
    if first in ("emergency", "stop"):
        return "emergency"
    if first in ("normal_exit", "walkrun_exit", "walkrun", "exit"):
        return "normal_exit"
    if first in ("damping_exit", "damping", "passive"):
        return "damping_exit"

    if first in ("action", "run", "start"):
        if len(tokens) < 2:
            raise ValueError("action requires a name, e.g. action kick_right")
        action = tokens[1]
        rest = tokens[2:]
    else:
        # Bare action name, e.g. "kick_right".
        action = tokens[0]
        rest = tokens[1:]

    parts = ["ACTION", action]
    parts.extend(_parse_timing_tokens(rest, default_delay_ms))
    return " ".join(parts)


def print_interactive_help(default_delay_ms: int) -> None:
    delay_note = (
        f"默认裸动作延迟 {default_delay_ms}ms；输入 now/immediate 可立即执行。"
        if default_delay_ms >= 0 else
        "默认裸动作立即执行。"
    )
    print(
        f"""
可输入命令：
  kick_right                         发送动作，{delay_note}
  hlqk
  action kick_right delay_ms=500     延迟 500ms 执行动作
  action kick_right now              立即执行动作
  action kick_right start_ms=1730000000000
  emergency / stop                   停止动作，进入阻尼和 SHELL_IDLE，机器人程序继续运行
  normal_exit / walkrun / exit        切回 walkrun 并退出机器人 action_go_remote_test
  damping_exit / passive             进入 passive 阻尼并退出机器人 action_go_remote_test
  raw ACTION kick_right delay_ms=500  发送原始 UDP 字符串
  help                               显示帮助
  quit / q                           只退出上位机窗口，不给机器人发退出
"""
    )


def interactive_loop(args: argparse.Namespace) -> int:
    print("[action_go_udp_send] interactive mode")
    print(f"targets={args.host} port={args.port} repeat={args.repeat}")
    print_interactive_help(args.default_delay_ms)
    while True:
        try:
            line = input("action_go> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[action_go_udp_send] local quit")
            return 0
        if not line:
            continue
        try:
            message = build_interactive_message(line, args.default_delay_ms)
            if message == "__QUIT__":
                print("[action_go_udp_send] local quit")
                return 0
            if message == "__HELP__":
                print_interactive_help(args.default_delay_ms)
                continue
            send_udp(args.host, args.port, message, args)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send UDP commands to action_go_remote_test on one or more robots."
    )
    parser.add_argument(
        "--host", "--target", "-H",
        action="append",
        default=None,
        help="目标 IP/广播地址/组播地址。可重复。默认 255.255.255.255",
    )
    parser.add_argument("--port", "-p", type=int, default=45678, help="UDP 端口，默认 45678")
    parser.add_argument("--broadcast", action="store_true", help="允许发送广播包")
    parser.add_argument("--ttl", type=int, default=1, help="组播 TTL，默认 1")
    parser.add_argument(
        "--cmd",
        choices=["action", "emergency", "normal_exit", "damping_exit"],
        default="action",
        help="指令类型，默认 action",
    )
    parser.add_argument("--action", "-a", help="动作名，例如 kick_right / hlqk")
    parser.add_argument("--delay-ms", type=int, help="机器人收到 UDP 后延迟多少毫秒执行")
    parser.add_argument("--start-ms", type=int, help="Unix 毫秒时间戳，到该时间执行；需要时间同步")
    parser.add_argument(
        "--start-after-ms",
        type=int,
        help="以上位机当前时间计算未来 Unix 毫秒时间戳；需要时间同步",
    )
    parser.add_argument("--raw", help="直接发送原始字符串，设置后忽略 --cmd/--action")
    parser.add_argument("--auth-key-file", help="HMAC-SHA256 共享密钥文件；远程/广播/组播控制必需")
    parser.add_argument("--auth-key-id", default="default", help="接收端配置的共享密钥标识")
    parser.add_argument("--repeat", type=int, default=1, help="重复发送次数，默认 1")
    parser.add_argument("--interval-ms", type=int, default=50, help="重复发送间隔，默认 50ms")
    parser.add_argument("--interactive", "-i", action="store_true", help="持续运行，在终端窗口内输入命令")
    parser.add_argument(
        "--default-delay-ms",
        type=int,
        default=500,
        help="交互模式下裸动作名的默认延迟；设为 -1 表示立即执行。默认 500",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印，不发送")
    args = parser.parse_args()

    if args.delay_ms is not None and args.delay_ms < 0:
        raise SystemExit("ERROR: --delay-ms must be >= 0")
    if args.start_after_ms is not None and args.start_after_ms < 0:
        raise SystemExit("ERROR: --start-after-ms must be >= 0")
    if sum(v is not None for v in [args.delay_ms, args.start_ms, args.start_after_ms]) > 1:
        raise SystemExit("ERROR: choose only one of --delay-ms / --start-ms / --start-after-ms")
    if args.repeat <= 0:
        raise SystemExit("ERROR: --repeat must be > 0")
    if args.interval_ms < 0:
        raise SystemExit("ERROR: --interval-ms must be >= 0")
    if args.default_delay_ms < -1:
        raise SystemExit("ERROR: --default-delay-ms must be >= -1")

    if not args.host:
        args.host = ["127.0.0.1"]
    if any(host not in ("127.0.0.1", "localhost") for host in args.host) and not args.auth_key_file:
        raise SystemExit("ERROR: non-loopback UDP control requires --auth-key-file")
    if args.auth_key_file:
        mode = os.stat(args.auth_key_file).st_mode
        if mode & 0o077:
            raise SystemExit("ERROR: --auth-key-file must not be accessible by group or other users")
    return args


def send_udp(hosts: List[str], port: int, message: str, args: argparse.Namespace) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if args.broadcast or any(looks_like_broadcast(h) for h in hosts):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if any(is_multicast(h) for h in hosts):
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, args.ttl)

        for i in range(args.repeat):
            outbound = sign_message(message, args)
            data = outbound.encode("utf-8")
            for host in hosts:
                if args.dry_run:
                    print(f"[dry-run] {host}:{port} <- {outbound}")
                else:
                    sock.sendto(data, (host, port))
                    print(f"sent {host}:{port} <- {outbound}")
            if i + 1 < args.repeat:
                time.sleep(args.interval_ms / 1000.0)
    finally:
        sock.close()


def main() -> int:
    args = parse_args()
    if args.interactive:
        return interactive_loop(args)
    message = build_message(args)
    send_udp(args.host, args.port, message, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
