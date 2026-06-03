#!/usr/bin/env python3
"""Stream D435i frames over a length-prefixed TCP socket.

Wire format: 4-byte little-endian payload length + JPEG-or-PNG bytes. Consumers
decode via opencv's magic-byte sniffing (cv::imdecode accepts both).

Default bind is 127.0.0.1 — the streamer + consumer normally run on the same
Orin. Pass `--bind 0.0.0.0` for offboard viewers (base station).

Routing: loopback clients (127.0.0.0/8) are served inline from the main
thread, so the onboard inference node sits inside the producer's tight loop
(frame pull + encode + sendall). Each non-loopback client gets its own sender
thread + 1-slot queue with drop-oldest-on-full semantics, so a slow or dead
offboard consumer can never stall the loopback path. Priority-by-construction
— no nice()/sched knobs.

Workflow (Orin):
    sudo /unitree/sbin/mscli stopservice video_hub_pc4   # release the camera
    python3 camera_streamer.py                            # JPEG, loopback only
    python3 camera_streamer.py --bind 0.0.0.0             # accept offboard too
    python3 camera_streamer.py --codec png -v             # PNG + per-sec fps log

Why bypass DDS at all:
    Unitree's videohub_pc4 publishes via CycloneDDS 0.7; the desktop's
    CycloneDDS 0.10 can't reassemble its fragmented messages, truncating each
    frame to ~360B (→ green screen). Direct TCP sidesteps that entirely. The
    Unitree video service holds the camera via master_service, which respawns
    it on pkill — only `mscli stopservice` actually releases it.
"""

import argparse
import queue
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


def _is_loopback(addr):
    return addr[0].startswith("127.")


class _RemoteClient:
    """Per-offboard-client sender. Owns a 1-slot queue + drain thread.

    Drop-oldest-on-full: live viewers want the latest frame, not a backlog,
    so under WiFi backpressure we discard the stale frame rather than letting
    a queue grow.
    """

    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.q = queue.Queue(maxsize=1)
        self.alive = True
        self.t = threading.Thread(target=self._drain, daemon=True)
        self.t.start()

    def offer(self, payload):
        try:
            self.q.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            pass

    def _drain(self):
        while self.alive:
            payload = self.q.get()
            if payload is None:
                break
            try:
                self.conn.sendall(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.alive = False
                break

    def close(self):
        self.alive = False
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        try:
            self.conn.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Stream D435i frames over TCP.")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="bind address (default: 127.0.0.1; use 0.0.0.0 for offboard)")
    ap.add_argument("--codec", choices=ENCODERS.keys(), default="jpeg",
                    help="frame codec (default: jpeg)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log per-second fps, payload size, client counts")
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

    loopback = []                       # list[socket] — served inline from main
    remotes = []                        # list[_RemoteClient] — each owns its thread
    clients_lock = threading.Lock()

    def accept_loop():
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if _is_loopback(addr):
                with clients_lock:
                    loopback.append(conn)
                print(f"Connected (loopback, inline): {addr}", flush=True)
            else:
                client = _RemoteClient(conn, addr)
                with clients_lock:
                    remotes.append(client)
                print(f"Connected (remote, threaded): {addr}", flush=True)

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
            lb_snapshot = list(loopback)
            rm_snapshot = list(remotes)

        # Inline (loopback): part of the producer's tight loop. sendall over lo
        # is a kernel memcpy — tens of µs at this bitrate, never blocks here.
        dead_lb = []
        for conn in lb_snapshot:
            try:
                conn.sendall(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead_lb.append(conn)

        # Decoupled (remote): non-blocking handoff; sender threads do the I/O.
        dead_rm = []
        for c in rm_snapshot:
            if c.alive:
                c.offer(payload)
            else:
                dead_rm.append(c)

        if dead_lb or dead_rm:
            with clients_lock:
                for conn in dead_lb:
                    try:
                        loopback.remove(conn)
                    except ValueError:
                        pass
                    try:
                        conn.close()
                    except OSError:
                        pass
                for c in dead_rm:
                    try:
                        remotes.remove(c)
                    except ValueError:
                        pass
                    c.close()
                lb_n, rm_n = len(loopback), len(remotes)
            for _ in dead_lb:
                print(f"Loopback client disconnected (remaining lb={lb_n}, rm={rm_n})",
                      flush=True)
            for c in dead_rm:
                print(f"Remote client {c.addr} disconnected "
                      f"(remaining lb={lb_n}, rm={rm_n})", flush=True)

        if args.verbose:
            log_count += 1
            now = time.time()
            if now - last_log >= 1.0:
                print(f"streamer: {log_count / (now - last_log):.1f} fps  "
                      f"codec={args.codec}  buf={len(buf) // 1024}KB  "
                      f"lb={len(loopback)} rm={len(remotes)}", flush=True)
                last_log, log_count = now, 0


if __name__ == "__main__":
    main()
