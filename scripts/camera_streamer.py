#!/usr/bin/env python3
"""Stream the D435i camera from the Orin to a desktop viewer over TCP.

Background:
    The Unitree videohub_pc4 service encodes D435i frames to H264 and publishes
    via bare DDS (CycloneDDS 0.7 on Foxy). The desktop runs CycloneDDS 0.10
    (Humble) which can't reassemble large fragmented messages from the 0.7
    publisher -- frames get truncated from ~40KB to ~360 bytes, producing a
    green screen. This script bypasses both the broken publisher and DDS
    entirely with direct TCP PNG streaming via pyrealsense2.

Workflow (on the Orin):
    1. Stop the Unitree video service that holds the camera (master_service
       supervises videohub_pc4 and will respawn it on pkill, so the supervisor
       must be told to stop it):
           sudo /unitree/sbin/mscli stopservice video_hub_pc4
    2. Run this streamer:
           python3 camera_streamer.py
    3. On the desktop, run the viewer:
           python3 view_front_camera.py <orin_ip>
"""

import argparse
import pyrealsense2 as rs
import cv2
import numpy as np
import socket
import struct
import subprocess
import threading
import signal
import sys
import time

WIDTH = 640
HEIGHT = 480
FPS = 60
PNG_COMPRESSION = 1  # 0=none, 9=max; 1 keeps encode CPU low (frames are lossless either way)
PORT = 5555


def cleanup_camera():
    """Stop Unitree videohub via master_service so it doesn't get respawned."""
    print("Stopping video_hub_pc4 via master_service...", flush=True)
    subprocess.run(["sudo", "/unitree/sbin/mscli", "stopservice", "video_hub_pc4"],
                   capture_output=True)
    time.sleep(2)
    print("Cleanup done.", flush=True)


def client_handler(conn, addr, clients, lock):
    """Kept alive just to detect disconnects via the send loop."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Stream D435i over TCP")
    parser.add_argument("--onboard", action="store_true",
                        help="Bind to 127.0.0.1 (streamer + node on same device)")
    args = parser.parse_args()

    bind_addr = "127.0.0.1" if args.onboard else "0.0.0.0"

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    cleanup_camera()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    print(f"Camera started: {WIDTH}x{HEIGHT} @ {FPS}fps", flush=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_addr, PORT))
    server.listen(5)
    print(f"Waiting for clients on {bind_addr}:{PORT}...", flush=True)

    clients = []
    clients_lock = threading.Lock()

    def accept_loop():
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Connected: {addr} (total: {len(clients) + 1})", flush=True)
            with clients_lock:
                clients.append(conn)

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()

    while True:
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        _, png = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        payload = struct.pack("<I", len(png)) + png.tobytes()

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


if __name__ == "__main__":
    main()
