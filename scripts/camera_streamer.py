#!/usr/bin/env python3
"""Stream D435i frames over a length-prefixed TCP socket.

Wire format: 4-byte little-endian payload length + payload. Without `--depth`
the payload is JPEG-or-PNG bytes and every byte is as it always was; with it,
the payload is the self-describing multi-plane record parsed by
`include/vision_encoders/frame_wire.hpp` (colour + depth in ONE record, so a
consumer never has to pair two streams by timestamp).

Depth ships RAW uint16 millimetres, already downscaled: sys1 classifies at
160 px and NEAREST-resizes depth to match, so 160x120 raw is 38 KB and 0 ms of
encode — smaller than the JPEG. PNG-16 at full res costs 10-25 ms on the Orin
and would blow the 60 Hz loop. `rs.align(color)` runs ONLY on the frames that
carry a depth plane (`--depth-hz`), for the same reason.

Intrinsics ride per-plane, in that plane's own pixels, so consumers need no
camera config and the depth downscale stays exact.

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
    python3 camera_streamer.py --depth                    # + depth for sys1

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

# frame_wire.hpp — keep the two in step; the C++ side is the reference.
MAGIC = b"VD1"
KIND_COLOR, KIND_DEPTH = 0, 1
FMT_RAW_U16_MM, FMT_JPEG, FMT_PNG = 0, 1, 2
_PLANE = struct.Struct("<BBHHffffI")


def _plane(kind, fmt, w, h, intr, payload):
    """One wire plane: fixed header + bytes. `intr` = (fx, fy, cx, cy)."""
    return _PLANE.pack(kind, fmt, w, h, *intr, len(payload)) + payload


def _scaled_intrinsics(intr, w, h):
    """Colour intrinsics expressed in a downscaled plane's own pixels."""
    sx, sy = w / intr.width, h / intr.height
    return (intr.fx * sx, intr.fy * sy, intr.ppx * sx, intr.ppy * sy)


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
    ap.add_argument("--depth", action="store_true",
                    help="attach a depth plane (multi-plane wire) — needed by sys1")
    ap.add_argument("--depth-size", default="160x120",
                    help="depth plane WxH; sys1 classifies at 160 px (default: 160x120)")
    ap.add_argument("--depth-hz", type=float, default=10.0,
                    help="depth plane rate; align() runs only on these frames (default: 10)")
    args = ap.parse_args()

    ext, encode_params = ENCODERS[args.codec]
    codec_fmt = FMT_JPEG if args.codec == "jpeg" else FMT_PNG
    dw, dh = (int(v) for v in args.depth_size.lower().split("x"))
    depth_period = FPS / max(args.depth_hz, 1e-6)

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    stop_unitree_video_service()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    if args.depth:
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    profile = pipe.start(cfg)
    color_intr = profile.get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color) if args.depth else None
    depth_scale_mm = (1000.0 * profile.get_device().first_depth_sensor().get_depth_scale()
                      if args.depth else 0.0)
    color_intrinsics = _scaled_intrinsics(color_intr, WIDTH, HEIGHT)
    depth_intrinsics = _scaled_intrinsics(color_intr, dw, dh)  # aligned to colour
    print(f"Camera started: {WIDTH}x{HEIGHT} @ {FPS}fps  codec={args.codec}"
          + (f"  depth={dw}x{dh} @ {args.depth_hz:g}Hz" if args.depth else ""), flush=True)

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

    last_log, log_count, frame_i, depth_due = time.time(), 0, 0, 0.0
    while True:
        frames = pipe.wait_for_frames()
        # Decimated: align() is the expensive half of a depth frame, so only pay
        # it on the frames that actually carry one.
        want_depth = args.depth and frame_i >= depth_due
        if want_depth:
            frames = align.process(frames)
            depth_due = frame_i + depth_period
        color = frames.get_color_frame()
        if not color:
            continue
        frame_i += 1
        img = np.asanyarray(color.get_data())
        _, buf = cv2.imencode(ext, img, encode_params)

        if not args.depth:
            body = buf.tobytes()                       # legacy wire, unchanged
        else:
            planes = _plane(KIND_COLOR, codec_fmt, WIDTH, HEIGHT,
                            color_intrinsics, buf.tobytes())
            depth = frames.get_depth_frame() if want_depth else None
            if depth:
                z = np.asanyarray(depth.get_data())    # uint16, sensor units
                z = cv2.resize(z, (dw, dh), interpolation=cv2.INTER_NEAREST)
                if abs(depth_scale_mm - 1.0) > 1e-6:
                    z = (z.astype(np.float32) * depth_scale_mm).astype(np.uint16)
                planes += _plane(KIND_DEPTH, FMT_RAW_U16_MM, dw, dh,
                                 depth_intrinsics, np.ascontiguousarray(z).tobytes())
            body = MAGIC + bytes([1 + bool(depth)]) + planes
        payload = struct.pack("<I", len(body)) + body

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
