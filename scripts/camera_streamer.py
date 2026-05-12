#!/usr/bin/env python3
"""Stream D435i frames over a length-prefixed TCP socket.

Wire format: 4-byte little-endian payload length + JPEG-or-PNG bytes. Consumers
decode via opencv's magic-byte sniffing (cv::imdecode accepts both).

Default bind is 127.0.0.1 — the streamer + consumer normally run on the same
Orin. Pass `--bind 0.0.0.0` for remote viewers (offboard dev only).

Workflow (Orin):
    sudo /unitree/sbin/mscli stopservice video_hub_pc4   # release the camera
    python3 camera_streamer.py                            # JPEG, loopback
    python3 camera_streamer.py --codec png -v             # PNG + per-sec fps log
    python3 camera_streamer.py --bind 0.0.0.0             # accept remote clients

Why bypass DDS at all:
    Unitree's videohub_pc4 publishes via CycloneDDS 0.7; the desktop's
    CycloneDDS 0.10 can't reassemble its fragmented messages, truncating each
    frame to ~360B (→ green screen). Direct TCP sidesteps that entirely. The
    Unitree video service holds the camera via master_service, which respawns
    it on pkill — only `mscli stopservice` actually releases it.
"""

import argparse
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 60
PORT = 5555

# JPEG: ~50-100 KB/frame, ~2-5 ms encode on Orin (60 Hz capable).
# PNG:  ~600-700 KB/frame, ~25-30 ms encode on Orin (~23 Hz ceiling).
ENCODERS = {
    "jpeg": (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]),
    "png":  (".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]),
}


def stop_unitree_video_service():
    print("Stopping video_hub_pc4 via master_service...", flush=True)
    subprocess.run(["sudo", "/unitree/sbin/mscli", "stopservice", "video_hub_pc4"],
                   capture_output=True)
    time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description="Stream D435i frames over TCP.")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="bind address (default: 127.0.0.1; use 0.0.0.0 for remote)")
    ap.add_argument("--codec", choices=ENCODERS.keys(), default="jpeg",
                    help="frame codec (default: jpeg)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log per-second fps, payload size, client count")
    args = ap.parse_args()

    ext, encode_params = ENCODERS[args.codec]

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    stop_unitree_video_service()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    print(f"Camera started: {WIDTH}x{HEIGHT} @ {FPS}fps  codec={args.codec}", flush=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, PORT))
    server.listen(5)
    print(f"Listening on {args.bind}:{PORT}", flush=True)

    clients = []
    clients_lock = threading.Lock()

    def accept_loop():
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with clients_lock:
                clients.append(conn)
            print(f"Connected: {addr} (total: {len(clients)})", flush=True)

    threading.Thread(target=accept_loop, daemon=True).start()

    last_log, log_count = time.time(), 0
    while True:
        color = pipe.wait_for_frames().get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        _, buf = cv2.imencode(ext, img, encode_params)
        payload = struct.pack("<I", len(buf)) + buf.tobytes()

        with clients_lock:
            dead = []
            for conn in clients:
                try:
                    conn.sendall(payload)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(conn)
            for conn in dead:
                clients.remove(conn)
                conn.close()
                print(f"Client disconnected (remaining: {len(clients)})", flush=True)

        if args.verbose:
            log_count += 1
            now = time.time()
            if now - last_log >= 1.0:
                print(f"streamer: {log_count / (now - last_log):.1f} fps  "
                      f"codec={args.codec}  buf={len(buf) // 1024}KB  "
                      f"clients={len(clients)}", flush=True)
                last_log, log_count = now, 0


if __name__ == "__main__":
    main()
