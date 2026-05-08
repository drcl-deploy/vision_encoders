#!/usr/bin/env python3
"""Interactive FPV dataset viewer.

Loads an IsaacLab fpv_dataset.pt + metadata.yaml exported by play.py and lets
you scrub through (env_id, timestep) to eyeball frame ↔ latent alignment.

Usage:
    python3 fpv_dataset_viewer.py --dataset /path/to/fpv_dataset_dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.widgets import Slider, TextBox


def load_dataset(dataset_dir: Path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dir containing fpv_dataset.pt + metadata.yaml")
    args = ap.parse_args()

    frames, feats, meta = load_dataset(args.dataset)
    N, T, H, W, _ = frames.shape
    feat_dim = feats.shape[-1]
    enc = meta.get("encoder", {})
    num_patches = int(enc.get("num_patches", 6))
    latent_dim = int(enc.get("latent_dim", feat_dim // num_patches))
    assert num_patches * latent_dim == feat_dim, \
        f"num_patches({num_patches}) * latent_dim({latent_dim}) != feat_dim({feat_dim})"

    # symmetric color limits across the full dataset for stable visual comparison
    feat_abs_max = float(np.max(np.abs(feats)))

    state = {"env": 0, "t": 0, "guard": False}

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(
        f"fpv data viewer — task={meta.get('task','?')}  encoder={enc.get('model_name','?')}",
        fontsize=11,
    )

    # frame: small (32x57) — make it big-ish so pixels are visible
    ax_frame = fig.add_axes([0.04, 0.30, 0.30, 0.58])
    # latent heatmap: P x D, very wide → use full remaining width
    ax_lat = fig.add_axes([0.38, 0.30, 0.50, 0.58])
    cax_lat = fig.add_axes([0.90, 0.30, 0.015, 0.58])
    ax_stats = fig.add_axes([0.04, 0.18, 0.92, 0.08])
    ax_stats.axis("off")

    ax_frame.set_title(f"frame  {H}×{W}×3 uint8 RGB")
    im_frame = ax_frame.imshow(frames[0, 0], interpolation="nearest")
    ax_frame.set_xticks([])
    ax_frame.set_yticks([])

    ax_lat.set_title(f"latent  {num_patches} patches × {latent_dim} dim")
    lat0 = feats[0, 0].reshape(num_patches, latent_dim)
    im_lat = ax_lat.imshow(lat0, aspect="auto", cmap="RdBu_r",
                           vmin=-feat_abs_max, vmax=feat_abs_max,
                           interpolation="nearest")
    ax_lat.set_xlabel("hidden dim")
    ax_lat.set_ylabel("patch idx")
    ax_lat.set_yticks(range(num_patches))
    fig.colorbar(im_lat, cax=cax_lat)

    stats_txt = ax_stats.text(0.0, 0.5, "", fontsize=10, family="monospace",
                              transform=ax_stats.transAxes, va="center")

    def render():
        n, t = state["env"], state["t"]
        fr = frames[n, t]
        ft = feats[n, t].reshape(num_patches, latent_dim)
        im_frame.set_data(fr)
        im_lat.set_data(ft)
        stats_txt.set_text(
            f"env = {n} / {N - 1}      t = {t} / {T - 1}\n"
            f"frame  : min={int(fr.min()):3d}  max={int(fr.max()):3d}  "
            f"mean={fr.mean():6.2f}\n"
            f"latent : min={ft.min():+.4f}  max={ft.max():+.4f}  "
            f"mean={ft.mean():+.4f}  std={ft.std():.4f}  ‖·‖={np.linalg.norm(ft):.3f}"
        )
        fig.canvas.draw_idle()

    # --- widgets row at the bottom ---
    ax_env_label = fig.add_axes([0.04, 0.06, 0.06, 0.05]); ax_env_label.axis("off")
    ax_env_label.text(0, 0.5, f"env_id  (0..{N-1})", va="center")
    ax_env_box = fig.add_axes([0.13, 0.06, 0.05, 0.05])
    tb_env = TextBox(ax_env_box, "", initial="0")

    ax_t_slider = fig.add_axes([0.22, 0.07, 0.55, 0.03])
    sl_t = Slider(ax_t_slider, "t", 0, max(T - 1, 1), valinit=0, valstep=1)
    ax_t_box = fig.add_axes([0.82, 0.06, 0.06, 0.05])
    tb_t = TextBox(ax_t_box, "", initial="0")

    def on_env_submit(text):
        try:
            n = int(text)
        except ValueError:
            tb_env.set_val(str(state["env"]))
            return
        if not (0 <= n < N):
            tb_env.set_val(str(state["env"]))
            return
        state["env"] = n
        render()

    def on_slider_change(val):
        if state["guard"]:
            return
        t = int(val)
        if t == state["t"]:
            return
        state["t"] = t
        state["guard"] = True
        tb_t.set_val(str(t))
        state["guard"] = False
        render()

    def on_t_submit(text):
        if state["guard"]:
            return
        try:
            t = int(text)
        except ValueError:
            tb_t.set_val(str(state["t"]))
            return
        if not (0 <= t < T):
            tb_t.set_val(str(state["t"]))
            return
        state["t"] = t
        state["guard"] = True
        sl_t.set_val(t)
        state["guard"] = False
        render()

    tb_env.on_submit(on_env_submit)
    sl_t.on_changed(on_slider_change)
    tb_t.on_submit(on_t_submit)

    print(f"[viewer] dataset shape: frames={frames.shape} features={feats.shape}")
    print(f"[viewer] N={N} envs, T={T} steps, encoder={enc.get('model_name')}")
    render()
    plt.show()


if __name__ == "__main__":
    main()
