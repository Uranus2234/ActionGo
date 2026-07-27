#!/usr/bin/env python3
"""Send TWIST mimic_obs packets to generic_motion_tracking_player realtime inputs.

This is a protocol example for integrating a Nokov retargeter: replace
`default_mimic_obs()` with the 33-float retargeted frame from Nokov.

Transports:
  udp:   player listens with --realtime-transport udp --udp-port 15110
  zmq:   player SUB binds by default; this script PUB connects
  redis: script SETs the configured Redis key at Hz
"""
import argparse
import json
import socket
import time


def default_mimic_obs():
    return [
        0.793,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.2, 0.0, 1.2, 0.0,
        0.0, -0.2, 0.0, 1.2, 0.0,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["udp", "zmq", "redis"], default="udp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=15110)
    ap.add_argument("--zmq-endpoint", default="tcp://127.0.0.1:15111")
    ap.add_argument("--redis-key", default="action_mimic_g1")
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    dt = 1.0 / args.hz
    packet_text = json.dumps({"mimic_obs": default_mimic_obs()}, separators=(",", ":"))

    if args.transport == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Sending TWIST mimic_obs to udp://{args.host}:{args.port} at {args.hz:g} Hz")
        while True:
            sock.sendto(packet_text.encode(), (args.host, args.port))
            time.sleep(dt)

    if args.transport == "zmq":
        import zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.connect(args.zmq_endpoint)
        print(f"Publishing TWIST mimic_obs to ZMQ {args.zmq_endpoint} at {args.hz:g} Hz")
        # Avoid PUB/SUB slow-joiner dropping only the first packet.
        time.sleep(0.3)
        while True:
            sock.send_string(packet_text)
            time.sleep(dt)

    if args.transport == "redis":
        import redis
        r = redis.Redis(host=args.host, port=args.port, db=0)
        print(f"SET redis://{args.host}:{args.port}/{args.redis_key} at {args.hz:g} Hz")
        while True:
            r.set(args.redis_key, packet_text)
            time.sleep(dt)


if __name__ == "__main__":
    main()
