#!/usr/bin/env python3
"""Zoo bench: spawn encoder_node per artifact, measure, dump, plot.

For each model (default: every *.manifest.json in the package's models/):
serve synthetic frames over the camera TCP wire at --fps, run the node for
--duration seconds, and collect

    tokens_hz decode_ms infer_ms  (from the ImageTokens msg)
    cpu_pct rss_mb                (/proc/<pid>)
    gpu_util_pct gpu_mem_mb       (pynvml; skipped if unavailable)

-> <out>/bench_<ts>.csv + <out>/bench_<ts>.png (metric bars vs models).

Usage (workspace sourced, sim/streamer NOT needed — the bench is its own producer):
    python3 tests/fps.py
    python3 tests/fps.py --models theia-tiny dinov2-s --duration 20
"""

import argparse
import csv
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import wire  # noqa: E402

from vision_encoders.msg import ImageTokens  # noqa: E402

BENCH_PORT = 5599
BENCH_TOPIC = '/bench/tokens'
RAW_W, RAW_H = 640, 480


def _share_dir():
    out = subprocess.run(['ros2', 'pkg', 'prefix', 'vision_encoders'],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip()) / 'share' / 'vision_encoders'


def _node_binary():
    out = subprocess.run(['ros2', 'pkg', 'prefix', 'vision_encoders'],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip()) / 'lib' / 'vision_encoders' / 'encoder_node'


class FrameServer(threading.Thread):
    """Synthetic producer on the camera wire: moving gradient, JPEG q85 @ fps."""

    def __init__(self, fps):
        super().__init__(daemon=True)
        self.fps = fps
        self.stop_evt = threading.Event()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', BENCH_PORT))
        self.server.listen(2)
        self.server.settimeout(0.5)

    def _accept_loop(self, conns, lock):
        while not self.stop_evt.is_set():
            try:
                conn, _ = self.server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with lock:
                    conns.append(conn)
            except (socket.timeout, OSError):
                pass

    @staticmethod
    def _payloads(n=32):
        """Pre-encoded JPEG cycle — the producer must never be the bottleneck."""
        xx, yy = np.meshgrid(np.arange(RAW_W), np.arange(RAW_H))
        out = []
        for i in range(n):
            phase = i * 256 // n
            img = np.stack([
                ((xx + phase) % 256), ((yy + phase) % 256), ((xx + yy) % 256),
            ], axis=-1).astype(np.uint8)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            assert ok
            out.append(struct.pack('<I', len(buf)) + buf.tobytes())
        return out

    def run(self):
        payloads = self._payloads()
        conns, lock = [], threading.Lock()
        threading.Thread(target=self._accept_loop, args=(conns, lock), daemon=True).start()
        next_t = time.monotonic()
        i = 0
        while not self.stop_evt.is_set():
            payload = payloads[i % len(payloads)]
            i += 1
            with lock:
                snapshot = list(conns)
            dead = []
            for c in snapshot:
                try:
                    c.sendall(payload)
                except OSError:
                    dead.append(c)
            if dead:
                with lock:
                    for c in dead:
                        conns.remove(c)
            next_t += 1.0 / self.fps  # absolute schedule: no per-frame drift
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
        with lock:
            for c in conns:
                c.close()
        self.server.close()


class TokenSink(Node):
    def __init__(self):
        super().__init__('bench_sink')
        self.stamps, self.decode_ms, self.infer_ms = [], [], []
        self.create_subscription(ImageTokens, BENCH_TOPIC, self._on_msg, 10)

    def _on_msg(self, msg):
        self.stamps.append(time.monotonic())
        self.decode_ms.append(msg.decode_ms)
        self.infer_ms.append(msg.infer_ms)

    def reset(self):
        self.stamps, self.decode_ms, self.infer_ms = [], [], []


def _proc_sample(pid):
    """(cpu_jiffies, rss_mb) — caller diffs jiffies over wall time."""
    with open(f'/proc/{pid}/stat') as f:
        parts = f.read().split()
    jiffies = int(parts[13]) + int(parts[14])  # utime + stime
    rss_mb = 0.0
    with open(f'/proc/{pid}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                rss_mb = int(line.split()[1]) / 1024.0
                break
    return jiffies, rss_mb


def _gpu_sample(pid, nvml):
    if nvml is None:
        return None, None
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
    mem = None
    try:
        for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if p.pid == pid and p.usedGpuMemory:
                mem = p.usedGpuMemory / 1e6
    except nvml.NVMLError:
        pass
    return float(util), mem


def bench_model(tag, sink, duration, nvml):
    proc = subprocess.Popen(
        [str(_node_binary()), '--ros-args',
         '-p', f'model:={tag}',
         '-p', f'camera_port:={BENCH_PORT}',
         '-p', f'tokens_topic:={BENCH_TOPIC}',
         '-p', 'publish_frame:=false',
         '-p', 'inference_rate_hz:=1000.0'],  # uncapped: measure the model, not the tick
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        sink.reset()
        deadline = time.monotonic() + 90
        while not sink.stamps:  # session load + first token
            rclpy.spin_once(sink, timeout_sec=0.2)
            if proc.poll() is not None:
                raise RuntimeError(f'{tag}: node exited early (rc={proc.returncode})')
            if time.monotonic() > deadline:
                raise RuntimeError(f'{tag}: no tokens within 90 s')

        sink.reset()
        j0, _ = _proc_sample(proc.pid)
        t0 = time.monotonic()
        gpu_utils, gpu_mems, rss = [], [], 0.0
        while time.monotonic() - t0 < duration:
            rclpy.spin_once(sink, timeout_sec=0.05)
            if int((time.monotonic() - t0) * 2) > len(gpu_utils):  # ~2 Hz sampling
                _, rss = _proc_sample(proc.pid)
                u, m = _gpu_sample(proc.pid, nvml)
                if u is not None:
                    gpu_utils.append(u)
                if m is not None:
                    gpu_mems.append(m)
        j1, rss = _proc_sample(proc.pid)
        wall = time.monotonic() - t0
        hz = os.sysconf('SC_CLK_TCK')

        n = len(sink.stamps)
        return {
            'model': tag,
            'tokens_hz': round(n / wall, 1),
            'decode_ms': round(float(np.mean(sink.decode_ms)), 2) if n else None,
            'infer_ms': round(float(np.mean(sink.infer_ms)), 2) if n else None,
            'cpu_pct': round(100.0 * (j1 - j0) / hz / wall, 1),
            'rss_mb': round(rss, 1),
            'gpu_util_pct': round(float(np.mean(gpu_utils)), 1) if gpu_utils else None,
            'gpu_mem_mb': round(float(np.mean(gpu_mems)), 1) if gpu_mems else None,
        }
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def plot(rows, png_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = ['tokens_hz', 'infer_ms', 'cpu_pct', 'rss_mb', 'gpu_util_pct', 'gpu_mem_mb']
    models = [r['model'] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, metric in zip(axes.flat, metrics):
        vals = [r[metric] if r[metric] is not None else 0 for r in rows]
        ax.bar(models, vals, color='tab:orange')
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'encoder zoo bench — {datetime.now():%Y-%m-%d %H:%M}')
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f'plot -> {png_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', default=None,
                        help='artifact tags (default: every manifest in models/)')
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--fps', type=float, default=60.0)
    parser.add_argument('--out', default=str(Path(__file__).parent / 'bench'))
    args = parser.parse_args()

    models = args.models or sorted(
        p.name.replace('.manifest.json', '')
        for p in (_share_dir() / 'models').glob('*.manifest.json'))
    if not models:
        raise SystemExit('no artifacts found — export first (vibe: export-encoder)')

    try:
        import pynvml as nvml
        nvml.nvmlInit()
    except Exception:
        nvml = None
        print('pynvml unavailable — gpu columns skipped (Orin: watch tegrastats)')

    server = FrameServer(args.fps)
    server.start()
    rclpy.init()
    sink = TokenSink()

    rows = []
    try:
        for tag in models:
            print(f'--- {tag}')
            try:
                row = bench_model(tag, sink, args.duration, nvml)
                rows.append(row)
                print('    ' + '  '.join(f'{k}={v}' for k, v in row.items() if k != 'model'))
            except RuntimeError as e:
                print(f'    FAILED: {e}')
    finally:
        server.stop_evt.set()
        sink.destroy_node()
        rclpy.shutdown()

    if rows:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%b%d_%H%M')
        csv_path = out_dir / f'bench_{ts}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'csv  -> {csv_path}')
        plot(rows, out_dir / f'bench_{ts}.png')


if __name__ == '__main__':
    main()
