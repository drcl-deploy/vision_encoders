#!/usr/bin/env python3
"""Self-contained joystick-gated rosbag2 recorder (one-shot, auto-timestamped).

Replaces `ros2 bag record` + pause/resume services (which aren't reliably
available in humble's rosbag2 build). One process owns:
  - capture: subscribes to /theia_input_frame + /theia_embedding
  - gating:  subscribes to /joy AND /lowstate (Unitree wireless gamepad)
  - writing: rosbag2_py.SequentialWriter -> <output_dir>/<YYYYMonDD_HHMM>/

Writer is opened at boot ("pre-warmed") so the A-press path is just a
boolean flip — no SequentialWriter.open() latency. Total trigger gap
~10–25 ms ≈ 1 frame at 50 Hz.

One press cycle per launch:
    A        -> recording=True; the next frame/embed callback writes
    Y or R1  -> recording=False, writer dropped (destructor flushes +
                commits SQLite). Bag is sealed and readable immediately.

Subsequent A presses after stop are ignored — relaunch for the next rollout.
If the user never presses A and Ctrl-C's, the empty pre-warmed bag dir
is cleaned up on shutdown.
"""
from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rosbag2_py import (
    ConverterOptions,
    SequentialWriter,
    StorageOptions,
    TopicMetadata,
)
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float32MultiArray

try:
    from unitree_hg.msg import LowState
    HAS_LOWSTATE = True
except ImportError:
    HAS_LOWSTATE = False

# Bit positions in xKeySwitchUnion (wireless_remote bytes 2..3 as uint16 LE)
GP_R1, GP_A, GP_Y = 0, 8, 11

FRAME_TOPIC = "/theia_input_frame"
EMBED_TOPIC = "/theia_embedding"


class RecordingNode(Node):
    def __init__(self):
        super().__init__("theia_recording_node")

        self.declare_parameter("output_dir", "")
        base = str(self.get_parameter("output_dir").value)
        if not base:
            raise RuntimeError("'output_dir' parameter is required (record:=...)")

        ts = datetime.datetime.now().strftime("%Y%b%d_%H%M")
        # `base` is the project/scenario dir (e.g. .../largetable); each launch
        # drops a timestamped bag inside it. Parent dir auto-created.
        self.output_dir = Path(base) / ts
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.output_dir.exists():
            raise RuntimeError(f"bag dir already exists (collision?): {self.output_dir}")

        # Joystick mappings
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("joy_start_button", 0)        # A
        self.declare_parameter("joy_stop_buttons", [3, 5])   # Y, RB
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("gamepad_start_bit", GP_A)
        self.declare_parameter("gamepad_stop_bits", [GP_Y, GP_R1])

        self.joy_start = int(self.get_parameter("joy_start_button").value)
        self.joy_stops = [int(b) for b in self.get_parameter("joy_stop_buttons").value]
        self.gp_start = int(self.get_parameter("gamepad_start_bit").value)
        self.gp_stops = [int(b) for b in self.get_parameter("gamepad_stop_bits").value]

        # State
        self.recording = False
        self.recorded = False           # one-shot lockout after stop
        self._joy_prev: list[int] = []
        self._gp_prev = 0

        # Pre-warm the writer NOW so the A-press path is a boolean flip,
        # not a file-creation + sqlite-init round trip.
        self.writer: SequentialWriter | None = self._make_writer()

        # Data capture subscriptions
        self.create_subscription(Image, FRAME_TOPIC, self._on_frame, 10)
        self.create_subscription(Float32MultiArray, EMBED_TOPIC, self._on_embed, 10)

        # Joystick subscriptions
        joy_topic = str(self.get_parameter("joy_topic").value)
        self.create_subscription(Joy, joy_topic, self._on_joy, 10)
        srcs = [f"joy={joy_topic}"]
        if HAS_LOWSTATE:
            ls_topic = str(self.get_parameter("lowstate_topic").value)
            self.create_subscription(LowState, ls_topic, self._on_lowstate, 10)
            srcs.append(f"lowstate={ls_topic}")
        else:
            self.get_logger().warn("unitree_hg not importable — /joy only")

        self.get_logger().info(
            f"recording_node up; bag will be written to {self.output_dir}/  "
            f"sources=[{', '.join(srcs)}]  start=A  stop=[Y, R1]"
        )

    # -----------------------------------------------------------------
    def _make_writer(self) -> SequentialWriter:
        writer = SequentialWriter()
        writer.open(
            StorageOptions(uri=str(self.output_dir), storage_id="sqlite3"),
            ConverterOptions(input_serialization_format="cdr",
                             output_serialization_format="cdr"),
        )
        writer.create_topic(TopicMetadata(
            name=FRAME_TOPIC,
            type="sensor_msgs/msg/Image",
            serialization_format="cdr",
        ))
        writer.create_topic(TopicMetadata(
            name=EMBED_TOPIC,
            type="std_msgs/msg/Float32MultiArray",
            serialization_format="cdr",
        ))
        return writer

    def _start(self, source: str):
        self.recording = True
        self.get_logger().info(f"-> START ▶ recording to {self.output_dir}  ({source})")

    def _stop(self, source: str):
        # Drop the only reference; CPython refcount hits 0 and the C++
        # destructor flushes + commits the sqlite db immediately.
        self.recording = False
        self.recorded = True
        self.writer = None
        self.get_logger().info(
            f"-> STOP ■ bag closed and ready: {self.output_dir}  ({source})"
        )

    def _trigger(self, action: str, source: str):
        if action == "start" and not self.recording and not self.recorded:
            self._start(source)
        elif action == "start" and self.recorded:
            self.get_logger().warn(
                f"recording already finished at {self.output_dir} — "
                f"relaunch with a fresh record:= for the next rollout"
            )
        elif action == "stop" and self.recording:
            self._stop(source)

    # -----------------------------------------------------------------
    # data callbacks (write only when recording)
    # -----------------------------------------------------------------
    def _on_frame(self, msg: Image):
        if self.recording and self.writer is not None:
            t = self.get_clock().now().nanoseconds
            self.writer.write(FRAME_TOPIC, serialize_message(msg), t)

    def _on_embed(self, msg: Float32MultiArray):
        if self.recording and self.writer is not None:
            t = self.get_clock().now().nanoseconds
            self.writer.write(EMBED_TOPIC, serialize_message(msg), t)

    # -----------------------------------------------------------------
    # joystick callbacks
    # -----------------------------------------------------------------
    def _on_joy(self, msg: Joy):
        buttons = list(msg.buttons)

        def rising(idx: int) -> bool:
            return (idx < len(buttons) and idx < len(self._joy_prev)
                    and buttons[idx] == 1 and self._joy_prev[idx] == 0)

        if rising(self.joy_start):
            self._trigger("start", "joy")
        else:
            for b in self.joy_stops:
                if rising(b):
                    self._trigger("stop", "joy")
                    break
        self._joy_prev = buttons

    def _on_lowstate(self, msg):
        wr = msg.wireless_remote
        if len(wr) < 4:
            return
        curr = int(wr[2]) | (int(wr[3]) << 8)
        rising_bits = curr & ~self._gp_prev
        if rising_bits & (1 << self.gp_start):
            self._trigger("start", "gamepad")
        else:
            for b in self.gp_stops:
                if rising_bits & (1 << b):
                    self._trigger("stop", "gamepad")
                    break
        self._gp_prev = curr

    def destroy_node(self):
        if self.recording:
            # Ctrl-C mid-record: flush the partial bag.
            self._stop("shutdown")
        elif not self.recorded:
            # Pre-warmed bag was never used — drop the writer and clean up
            # the empty dir so we don't litter the filesystem.
            self.writer = None
            try:
                shutil.rmtree(self.output_dir)
                self.get_logger().info(f"removed unused empty bag: {self.output_dir}")
            except OSError as e:
                self.get_logger().warn(
                    f"failed to remove empty bag {self.output_dir}: {e}"
                )
        super().destroy_node()


def main():
    rclpy.init()
    try:
        node = RecordingNode()
    except RuntimeError as e:
        print(f"[recording_node] startup failed: {e}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
