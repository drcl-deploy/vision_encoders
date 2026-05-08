"""Implementation consistency: IsaacLab encoder vs theia_tiny_node (C++).

Goal
----
Given the same RGB frame fed to both pathways, are the resulting Theia-Tiny
patch embeddings the same?

  IsaacLab (sim, python) :  frame -> image_features_ext -> y_lab    (in dataset)
  theia_tiny_node (C++)  :  frame -> TCP -> imdecode -> resize ->
                            normalize -> ORT/CUDA -> y_node

This script:
  1. Loads the IsaacLab fpv_dataset.pt + metadata.yaml exported by
     isaaclab/.../play.py --export_fpv_dataset.
  2. Spawns theia_tiny_node as a subprocess, parameterized by
     config/theia_tiny_consistency_test.yaml — which forces
     raw_frame_* == model_input_* so both cv::resize calls are no-ops.
     Without that, interpolation corrupts the input and inflates MSE for
     reasons unrelated to the encoder.
  3. Acts as the camera TCP server on the configured port, mirroring
     camera_streamer.py's wire format: <uint32 LE length><PNG bytes>.
     PNG (lossless) so wire round-trip isn't a noise source.
  4. Subscribes to /theia_embedding, sends frames one-by-one, computes
     MSE / max-abs-err / cosine-sim against dataset features.

Usage
-----
    # one terminal — fully self-contained
    python3 implementation_consistency.py --dataset /path/to/fpv_dataset_dir
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_CONFIG = PKG_ROOT / "config" / "theia_tiny_consistency_test.yaml"


def load_test_config(config_path: Path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    p = cfg["theia_tiny_node"]["ros__parameters"]
    model_path = p["model_path"]
    if not os.path.isabs(model_path):
        model_path = str(PKG_ROOT / model_path)
    return {
        "model_path": model_path,
        "camera_ip": p.get("camera_ip", "127.0.0.1"),
        "camera_port": int(p.get("camera_port", 5555)),
        "raw_w": int(p["raw_frame_width"]),
        "raw_h": int(p["raw_frame_height"]),
        "model_w": int(p["model_input_width"]),
        "model_h": int(p["model_input_height"]),
    }


def spawn_node(config_path: Path, model_path: str) -> subprocess.Popen:
    """Launch theia_tiny_node with the test config; model_path overridden absolute."""
    cmd = [
        "ros2", "run", "vision_encoders", "theia_tiny_node",
        "--ros-args",
        "--params-file", str(config_path),
        "-p", f"model_path:={model_path}",
    ]
    print(f"[test] spawning node: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        # own process group so we can SIGINT the whole tree on cleanup
        preexec_fn=os.setsid,
    )


def kill_node(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=2)


def load_dataset(dataset_dir: Path):
    pt = torch.load(dataset_dir / "fpv_dataset.pt", map_location="cpu", weights_only=False)
    with open(dataset_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    frames = pt["frames"]    # [N, T, H, W, 3] uint8
    feats = pt["features"]   # [N, T, feat_dim] float32
    return frames, feats, meta


class EmbeddingCollector(Node):
    """Pulls one Float32MultiArray off /theia_embedding, blocking with timeout."""

    def __init__(self, topic: str = "/theia_embedding"):
        super().__init__("theia_consistency_test")
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stamp = 0.0
        self.create_subscription(Float32MultiArray, topic, self._cb, 50)

    def _cb(self, msg: Float32MultiArray):
        with self._lock:
            self._latest = np.asarray(msg.data, dtype=np.float32)
            self._stamp = time.monotonic()

    def wait_fresh(self, after: float, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._stamp > after and self._latest is not None:
                    return self._latest.copy()
            rclpy.spin_once(self, timeout_sec=0.01)
        return None


class FrameTcpServer:
    """Bind on (host, port), accept one client, send PNG-encoded frames on demand."""

    def __init__(self, host: str, port: int):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((host, port))
        self.s.listen(1)
        self.conn = None

    def accept(self, timeout: float = 30.0):
        self.s.settimeout(timeout)
        print(f"[test] waiting for theia_tiny_node to connect on {self.s.getsockname()} ...")
        self.conn, addr = self.s.accept()
        self.conn.settimeout(5.0)
        print(f"[test] node connected from {addr}")

    def send_frame(self, rgb: np.ndarray):
        # node uses cv::imdecode (BGR). Encode BGR PNG so colors round-trip cleanly.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("PNG encode failed")
        payload = buf.tobytes()
        self.conn.sendall(struct.pack("<I", len(payload)) + payload)

    def close(self):
        try:
            if self.conn:
                self.conn.close()
        finally:
            self.s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dir containing fpv_dataset.pt + metadata.yaml")
    ap.add_argument("--config", type=Path, default=DEFAULT_TEST_CONFIG,
                    help="ros params yaml for the spawned node "
                         "(default: config/theia_tiny_consistency_test.yaml)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap total frames evaluated (default: all)")
    ap.add_argument("--per-frame-timeout", type=float, default=2.0)
    ap.add_argument("--embedding-topic", default="/theia_embedding")
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"[test] config not found: {args.config}")
    test_cfg = load_test_config(args.config)
    print(f"[test] using config: {args.config}")
    print(f"[test] node will run at raw={test_cfg['raw_w']}x{test_cfg['raw_h']} "
          f"model={test_cfg['model_w']}x{test_cfg['model_h']}")

    frames, feats, meta = load_dataset(args.dataset)
    N, T, H, W, _ = frames.shape
    feat_dim = feats.shape[-1]
    print(f"[test] dataset: N={N} T={T} HxW={H}x{W} feat_dim={feat_dim}")
    print(f"[test] meta.encoder = {meta.get('encoder')}")

    # Hard guard: dataset and node config MUST agree on resolution, else the
    # node's resize stages will silently distort frames and the MSE becomes
    # meaningless.
    if (W, H) != (test_cfg["raw_w"], test_cfg["raw_h"]) or \
       (W, H) != (test_cfg["model_w"], test_cfg["model_h"]):
        sys.exit(f"[test] FATAL: resolution mismatch — dataset={W}x{H}, "
                 f"raw={test_cfg['raw_w']}x{test_cfg['raw_h']}, "
                 f"model={test_cfg['model_w']}x{test_cfg['model_h']}. "
                 f"Update {args.config} so all three match.")

    rclpy.init()
    collector = EmbeddingCollector(args.embedding_topic)
    # Bind first so the spawned node sees an open port on its first attempt.
    server = FrameTcpServer(test_cfg["camera_ip"], test_cfg["camera_port"])
    node_proc = spawn_node(args.config, test_cfg["model_path"])
    server.accept(timeout=60.0)

    total = N * T if args.max_frames is None else min(args.max_frames, N * T)
    se_sum = 0.0       # sum of squared errors (for MSE over all elements)
    abs_max = 0.0
    cos_sum = 0.0
    count = 0
    mismatched_dim = 0

    try:
        t0 = time.time()
        for n in range(N):
            for t in range(T):
                if count >= total:
                    break
                rgb = frames[n, t].numpy() if torch.is_tensor(frames) else frames[n, t]
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                y_lab = feats[n, t].numpy() if torch.is_tensor(feats) else feats[n, t]
                y_lab = np.asarray(y_lab, dtype=np.float32)

                t_send = time.monotonic()
                server.send_frame(rgb)
                y_node = collector.wait_fresh(after=t_send, timeout=args.per_frame_timeout)
                if y_node is None:
                    print(f"[test] WARN: no embedding within {args.per_frame_timeout}s "
                          f"at (env={n}, t={t}); aborting.")
                    sys.exit(2)
                if y_node.shape[0] != y_lab.shape[0]:
                    mismatched_dim += 1
                    if mismatched_dim == 1:
                        print(f"[test] FATAL: shape mismatch — node={y_node.shape}, "
                              f"isaaclab={y_lab.shape}. Check num_patches/hidden_dim.")
                    sys.exit(3)

                err = y_node - y_lab
                se_sum += float(np.dot(err, err))
                abs_max = max(abs_max, float(np.max(np.abs(err))))
                denom = float(np.linalg.norm(y_node) * np.linalg.norm(y_lab) + 1e-12)
                cos_sum += float(np.dot(y_node, y_lab)) / denom
                count += 1

                if count % 50 == 0:
                    rate = count / max(1e-6, time.time() - t0)
                    print(f"[test] {count}/{total}  ({rate:.1f} fps)  "
                          f"running MSE={se_sum / (count * feat_dim):.3e}  "
                          f"max|e|={abs_max:.3e}  cos={cos_sum / count:.6f}")
            if count >= total:
                break
    finally:
        server.close()
        collector.destroy_node()
        rclpy.shutdown()
        kill_node(node_proc)

    if count == 0:
        print("[test] no frames evaluated.")
        sys.exit(1)

    mse = se_sum / (count * feat_dim)
    print("\n========== implementation consistency ==========")
    print(f"  frames evaluated : {count}")
    print(f"  per-element MSE  : {mse:.6e}")
    print(f"  max abs error    : {abs_max:.6e}")
    print(f"  mean cosine sim  : {cos_sum / count:.8f}")
    print("================================================")


if __name__ == "__main__":
    main()
