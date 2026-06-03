#!/usr/bin/env python3
"""Interactive domain comparison: IsaacLab fpv dataset vs C++ node rollout bag.

Left half:  source domain (IsaacLab fpv_dataset.pt) — env_id + t scrubbers
Right half: target domain (rosbag2 of /theia_input_frame + /theia_embedding,
            from `ros2 launch vision_encoders theia_tiny.launch.py record:=...`)
            — single rollout, only t scrubber

Bottom: latent_diff = latent_src - latent_tgt heatmap + scalar metrics.
Both halves drive the diff independently — pick any (env_src, t_src) vs any t_tgt.

Usage:
    python3 domain_comparison.py \\
        --source /path/to/fpv_dataset_dir \\
        --target /tmp/theia_dump
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.widgets import Slider, TextBox

# ROS bag reading
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# Global colormap for ALL heatmaps (src latent, tgt latent, diff). Tweak here.
# Good options for signed data: "RdBu_r", "coolwarm", "seismic", "PiYG", "PuOr",
# "BrBG", "twilight_shifted". For non-signed/abs view: "viridis", "magma".
COLORMAP = "RdBu_r"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_source(dataset_dir: Path):
    """IsaacLab fpv_dataset.pt + metadata.yaml."""
    pt = torch.load(dataset_dir / "fpv_dataset.pt", map_location="cpu", weights_only=False)
    with open(dataset_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    frames = pt["frames"]
    feats = pt["features"]
    if torch.is_tensor(frames):
        frames = frames.numpy()
    if torch.is_tensor(feats):
        feats = feats.numpy()
    return frames, feats, meta


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def read_bag(bag_path: Path,
             frame_topic: str = "/theia_input_frame",
             embed_topic: str = "/theia_embedding"):
    """Walk a rosbag2 dir, return per-topic message lists.

    Returns:
        frames:  list of (header_stamp_ns, H, W, rgb_uint8 array)
        embeds:  list of (recv_stamp_ns, flat_float32 array)
    """
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id=""),  # auto-detect
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if frame_topic not in topic_types or embed_topic not in topic_types:
        raise RuntimeError(
            f"Bag is missing required topics. Found: {list(topic_types)}; "
            f"need {frame_topic} and {embed_topic}."
        )

    frames, embeds = [], []
    while reader.has_next():
        topic, raw, recv_t = reader.read_next()
        if topic == frame_topic:
            msg = deserialize_message(raw, Image)
            if msg.encoding != "rgb8":
                raise RuntimeError(
                    f"{frame_topic} encoding is '{msg.encoding}', expected 'rgb8'. "
                    "Did you record from a node with publish_input_frame=true?"
                )
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                msg.height, msg.width, 3).copy()
            frames.append((_stamp_ns(msg.header.stamp), arr))
        elif topic == embed_topic:
            msg = deserialize_message(raw, Float32MultiArray)
            embeds.append((int(recv_t), np.asarray(msg.data, dtype=np.float32)))
    return frames, embeds


def pair_by_stamp(frames, embeds, tol_ns: int = 100_000_000):
    """Stamp-match each frame to its closest embedding (within tol).

    Frame stamp comes from header.stamp (set in infer_latest() right before publish).
    Embedding has no header — use rosbag's recv timestamp as its anchor. Both come
    from the same infer_latest() call, so they're typically <1ms apart.
    """
    if not frames or not embeds:
        return []
    embed_ts = np.asarray([e[0] for e in embeds], dtype=np.int64)
    paired = []
    used = set()
    for f_t, rgb in frames:
        i = int(np.argmin(np.abs(embed_ts - f_t)))
        if abs(embed_ts[i] - f_t) > tol_ns:
            continue
        if i in used:
            continue  # already paired — shouldn't happen for clean 1:1 streams
        used.add(i)
        paired.append((rgb, embeds[i][1]))
    return paired


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(src_frames, src_feats, src_meta, tgt_pairs):
    N, T_src, H_src, W_src, _ = src_frames.shape
    feat_dim = src_feats.shape[-1]
    enc = src_meta.get("encoder", {})
    P = int(enc.get("num_patches", 6))
    D = int(enc.get("latent_dim", feat_dim // P))
    assert P * D == feat_dim, f"P({P})*D({D}) != feat_dim({feat_dim})"

    T_tgt = len(tgt_pairs)
    if T_tgt == 0:
        raise RuntimeError("Target rollout is empty — no (frame, embedding) pairs after matching.")

    tgt_feat_dim = tgt_pairs[0][1].shape[0]
    if tgt_feat_dim != feat_dim:
        raise RuntimeError(
            f"Target embedding dim ({tgt_feat_dim}) != source ({feat_dim}). "
            "Encoder output shape mismatch — re-export ONNX or re-record bag."
        )
    H_tgt, W_tgt = tgt_pairs[0][0].shape[:2]

    # Symmetric color limits, shared across src+tgt+diff so values are visually comparable.
    src_abs = float(np.max(np.abs(src_feats)))
    tgt_abs = float(max(np.max(np.abs(p[1])) for p in tgt_pairs))
    feat_lim = max(src_abs, tgt_abs)
    diff_lim = feat_lim  # same scale as the latent panels — easy eye comparison

    state = {"env": 0, "t_src": 0, "t_tgt": 0, "guard": False}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"domain comparison — encoder={enc.get('model_name','?')}  "
        f"task={src_meta.get('task','?')}",
        fontsize=11,
    )

    # ----- row 1: frames + latents (src on left, tgt on right) -----
    # h1 sized to roughly match the rendered frame height (W/H aspect ~1.78
    # at frame box width 0.13, figsize (16,10)) so latent panels visually
    # align with the frame instead of towering over it.
    y1, h1 = 0.78, 0.13
    ax_frame_s = fig.add_axes([0.030, y1, 0.13, h1])
    ax_lat_s   = fig.add_axes([0.190, y1, 0.22, h1])
    cax_lat_s  = fig.add_axes([0.420, y1, 0.010, h1])
    ax_frame_t = fig.add_axes([0.500, y1, 0.13, h1])
    ax_lat_t   = fig.add_axes([0.660, y1, 0.22, h1])
    cax_lat_t  = fig.add_axes([0.890, y1, 0.010, h1])

    ax_frame_s.set_title(f"SRC frame  {H_src}×{W_src}", fontsize=10)
    ax_frame_t.set_title(f"TGT frame  {H_tgt}×{W_tgt}", fontsize=10)
    ax_lat_s.set_title(f"SRC latent  {P}×{D}", fontsize=10)
    ax_lat_t.set_title(f"TGT latent  {P}×{D}", fontsize=10)

    im_frame_s = ax_frame_s.imshow(src_frames[0, 0], interpolation="nearest")
    im_frame_t = ax_frame_t.imshow(tgt_pairs[0][0], interpolation="nearest")
    for ax in (ax_frame_s, ax_frame_t):
        ax.set_xticks([]); ax.set_yticks([])

    lat_s0 = src_feats[0, 0].reshape(P, D)
    lat_t0 = tgt_pairs[0][1].reshape(P, D)
    im_lat_s = ax_lat_s.imshow(lat_s0, aspect="auto", cmap=COLORMAP,
                               vmin=-feat_lim, vmax=feat_lim, interpolation="nearest")
    im_lat_t = ax_lat_t.imshow(lat_t0, aspect="auto", cmap=COLORMAP,
                               vmin=-feat_lim, vmax=feat_lim, interpolation="nearest")
    for ax in (ax_lat_s, ax_lat_t):
        ax.set_xlabel("hidden dim"); ax.set_ylabel("patch")
        ax.set_yticks(range(P))
    fig.colorbar(im_lat_s, cax=cax_lat_s)
    fig.colorbar(im_lat_t, cax=cax_lat_t)

    # ----- row 2: diff heatmap (full width) -----
    ax_diff = fig.add_axes([0.060, 0.34, 0.84, 0.22])
    cax_diff = fig.add_axes([0.910, 0.34, 0.012, 0.22])
    im_diff = ax_diff.imshow(lat_s0 - lat_t0, aspect="auto", cmap=COLORMAP,
                             vmin=-diff_lim, vmax=diff_lim, interpolation="nearest")
    ax_diff.set_title("latent_diff = src − tgt", fontsize=10)
    ax_diff.set_xlabel("hidden dim"); ax_diff.set_ylabel("patch")
    ax_diff.set_yticks(range(P))
    fig.colorbar(im_diff, cax=cax_diff)
    # marker for argmax-|diff| cell
    argmax_marker, = ax_diff.plot([], [], "k*", markersize=14, markerfacecolor="yellow",
                                  markeredgewidth=1.5)

    # ----- row 3: stats panel -----
    ax_stats = fig.add_axes([0.04, 0.16, 0.92, 0.14])
    ax_stats.axis("off")
    stats_txt = ax_stats.text(0.0, 1.0, "", fontsize=9, family="monospace",
                              transform=ax_stats.transAxes, va="top")

    # ----- row 4: controls -----
    # left half — source: env_id box + t_src slider + t_src box
    ax_env_lbl = fig.add_axes([0.030, 0.07, 0.06, 0.04]); ax_env_lbl.axis("off")
    ax_env_lbl.text(0, 0.5, f"env_id (0..{N-1})", va="center", fontsize=9)
    ax_env_box = fig.add_axes([0.115, 0.07, 0.04, 0.04])
    tb_env = TextBox(ax_env_box, "", initial="0")

    ax_ts_slider = fig.add_axes([0.190, 0.085, 0.18, 0.025])
    sl_ts = Slider(ax_ts_slider, "t_src", 0, max(T_src - 1, 1), valinit=0, valstep=1)
    ax_ts_box = fig.add_axes([0.385, 0.07, 0.04, 0.04])
    tb_ts = TextBox(ax_ts_box, "", initial="0")

    # right half — target: t_tgt slider + box (no env_id)
    ax_tt_slider = fig.add_axes([0.500, 0.085, 0.34, 0.025])
    sl_tt = Slider(ax_tt_slider, "t_tgt", 0, max(T_tgt - 1, 1), valinit=0, valstep=1)
    ax_tt_box = fig.add_axes([0.860, 0.07, 0.04, 0.04])
    tb_tt = TextBox(ax_tt_box, "", initial="0")

    # ----- render -----
    def render():
        n, ts, tt = state["env"], state["t_src"], state["t_tgt"]
        fr_s = src_frames[n, ts]
        ft_s = src_feats[n, ts].reshape(P, D)
        fr_t, flat_t = tgt_pairs[tt]
        ft_t = flat_t.reshape(P, D)
        diff = ft_s - ft_t

        im_frame_s.set_data(fr_s)
        im_frame_t.set_data(fr_t)
        im_lat_s.set_data(ft_s)
        im_lat_t.set_data(ft_t)
        im_diff.set_data(diff)

        # argmax-|diff| cell
        i_max, j_max = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
        argmax_marker.set_data([j_max], [i_max])

        # per-patch L2 of diff (which patch carries the gap?)
        per_patch = np.linalg.norm(diff, axis=1)

        l2 = float(np.linalg.norm(diff))
        mse = float((diff ** 2).mean())
        denom = float(np.linalg.norm(ft_s) * np.linalg.norm(ft_t) + 1e-12)
        cos = float((ft_s.flatten() @ ft_t.flatten()) / denom)

        stats_txt.set_text(
            f"SRC  env={n}/{N-1}  t={ts}/{T_src-1}    "
            f"min={ft_s.min():+.3f}  max={ft_s.max():+.3f}  "
            f"mean={ft_s.mean():+.4f}  std={ft_s.std():.4f}  ‖·‖={np.linalg.norm(ft_s):.3f}\n"
            f"TGT             t={tt}/{T_tgt-1}    "
            f"min={ft_t.min():+.3f}  max={ft_t.max():+.3f}  "
            f"mean={ft_t.mean():+.4f}  std={ft_t.std():.4f}  ‖·‖={np.linalg.norm(ft_t):.3f}\n"
            f"DIFF (src−tgt)   per-elem MSE={mse:.4e}   ‖diff‖={l2:.4f}   cos(src,tgt)={cos:+.6f}\n"
            f"  argmax|diff| @ (patch={i_max}, dim={j_max})  value={diff[i_max, j_max]:+.4f}  "
            f"|·|={abs(diff[i_max, j_max]):.4f}\n"
            f"  per-patch ‖diff‖ = [" + "  ".join(f"{v:.3f}" for v in per_patch) + "]"
        )
        fig.canvas.draw_idle()

    # ----- callbacks -----
    def on_env(text):
        try:
            n = int(text)
        except ValueError:
            tb_env.set_val(str(state["env"])); return
        if not (0 <= n < N):
            tb_env.set_val(str(state["env"])); return
        state["env"] = n
        render()

    def on_ts_slider(val):
        if state["guard"]: return
        ts = int(val)
        if ts == state["t_src"]: return
        state["t_src"] = ts
        state["guard"] = True; tb_ts.set_val(str(ts)); state["guard"] = False
        render()

    def on_ts_box(text):
        if state["guard"]: return
        try:
            ts = int(text)
        except ValueError:
            tb_ts.set_val(str(state["t_src"])); return
        if not (0 <= ts < T_src):
            tb_ts.set_val(str(state["t_src"])); return
        state["t_src"] = ts
        state["guard"] = True; sl_ts.set_val(ts); state["guard"] = False
        render()

    def on_tt_slider(val):
        if state["guard"]: return
        tt = int(val)
        if tt == state["t_tgt"]: return
        state["t_tgt"] = tt
        state["guard"] = True; tb_tt.set_val(str(tt)); state["guard"] = False
        render()

    def on_tt_box(text):
        if state["guard"]: return
        try:
            tt = int(text)
        except ValueError:
            tb_tt.set_val(str(state["t_tgt"])); return
        if not (0 <= tt < T_tgt):
            tb_tt.set_val(str(state["t_tgt"])); return
        state["t_tgt"] = tt
        state["guard"] = True; sl_tt.set_val(tt); state["guard"] = False
        render()

    tb_env.on_submit(on_env)
    sl_ts.on_changed(on_ts_slider); tb_ts.on_submit(on_ts_box)
    sl_tt.on_changed(on_tt_slider); tb_tt.on_submit(on_tt_box)

    render()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="IsaacLab fpv dataset dir (fpv_dataset.pt + metadata.yaml)")
    ap.add_argument("--target", type=Path, required=True,
                    help="rosbag2 dir containing /theia_input_frame + /theia_embedding")
    ap.add_argument("--frame-topic", default="/theia_input_frame")
    ap.add_argument("--embed-topic", default="/theia_embedding")
    ap.add_argument("--pair-tol-ms", type=float, default=100.0,
                    help="max stamp gap to count as a paired (frame, embedding)")
    args = ap.parse_args()

    print(f"[domain] loading source: {args.source}")
    src_frames, src_feats, src_meta = load_source(args.source)
    print(f"[domain]   src: frames={src_frames.shape}  feats={src_feats.shape}")

    print(f"[domain] reading target bag: {args.target}")
    raw_frames, raw_embeds = read_bag(args.target, args.frame_topic, args.embed_topic)
    print(f"[domain]   bag: {len(raw_frames)} frames, {len(raw_embeds)} embeddings")
    pairs = pair_by_stamp(raw_frames, raw_embeds, tol_ns=int(args.pair_tol_ms * 1e6))
    print(f"[domain]   paired (within {args.pair_tol_ms:.0f} ms): {len(pairs)}")

    if len(pairs) == 0:
        raise SystemExit(
            "No (frame, embedding) pairs were matched. Bag may be missing topics, "
            "encoding may be wrong, or stamp tolerance is too tight."
        )

    build_ui(src_frames, src_feats, src_meta, pairs)


if __name__ == "__main__":
    main()
