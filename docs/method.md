# method — the encoder-zoo revamp

phase-2 of the ViBe deployment: frozen vision backbones as swappable,
self-describing onnx artifacts feeding a C++ inference node. companion to
[cpp_control/docs/trackers/custom_sonic.md](../../cpp_control/docs/trackers/custom_sonic.md)
(phase-1, the policy side) and `unitree_ros2/docs/vibe/vision.md` (the
whole-stack architecture).

## background

v1 of this package (`theia_tiny_node`) proved the pipeline: TCP frames
(sim & real, one wire format) → decode → resize → normalize → ORT/CUDA →
patch tokens on a ROS topic. it worked, but every piece of *model knowledge*
lived in the C++:

| v1 smell | consequence |
|---|---|
| patch grid `(H/16)*(W/16)` hardcoded | breaks every patch-14 backbone (dinov2/3) |
| ImageNet mean/std hardcoded | breaks the CLIP family (CLIP stats), siglip (0.5/0.5) |
| `num_patches` from yaml, `hidden_dim` from graph | shape split-brain — the drift class phase-1 killed |
| `Float32MultiArray`, no stamp | no staleness gate, no frame↔token association |
| per-tick tensor alloc | avoidable hot-loop churn |
| export params (`hf_name`, `opset`) declared as ROS params | export concerns leaking into runtime |

meanwhile the training side grew `vibe/scripts/bench/enc_zoo.py`: seven
frozen backbones (theia-tiny, tinyclip-39m, clip-b32, siglip2-b16, dinov2-s,
dinov3-splus, mobileclip2-s2) behind one adapter API — norm stats, patch
size, CLS/register handling, native-resolution forward all resolved per
backbone, in python, once.

## goal

1. image → [encoder] → patch tokens (P×D) + cls token (D), published **as-is**
   — no post-processing; the policy onnx's cross-attention does the math
   (consistent with `vibe/repose/mdp/observations.py::image_feature`).
2. n backbones, each a `{onnx, manifest}` artifact pair — swapping = swapping
   files, **zero code change** anywhere.
3. theia-tiny is priority 1 (current training champion); the zoo validates
   the infra.
4. a bench (`tests/fps.py`) that spawns the node per backbone and plots
   fps / latency / cpu / gpu / memory vs model.

## philosophy

same move as phase-1: **the exporter owns all model knowledge; the C++ is a
dumb executor of a self-describing artifact.**

- exporter lives in **vibe** and wraps `enc_zoo` directly — byte-parity with
  training/bench code *by construction*, zero mirrored code in this package.
- all preprocessing (cast, /255, BGR→RGB, HWC→CHW, per-model normalization)
  is **baked into the onnx graph**. the C++ preprocessing shrinks to
  decode → resize → memcpy. an entire class of C++/python mismatch becomes
  unrepresentable.
- one node binary, one launch (`model:=<tag>`), one viewer, one msg —
  nothing user-facing carries a model name; the manifest is the model config.
- fail loud at startup (manifest vs graph dims), never mid-run.

## design rationale

### the artifact — vision.onnx.v1

```
wire frame (BGR HWC uint8 — exactly what cv::imdecode yields)
        │ cv::resize to manifest (H, W)      ← the only op left in C++
        ▼
┌─ <tag>.onnx ──────────────────────────────────────────────┐
│ input  image_bgr_hwc  (1, H, W, 3) uint8                  │
│   cast → /255 → BGR→RGB → HWC→CHW → (x − mean)/std        │  baked
│   backbone (native res, interpolate_pos_encoding)         │
│ outputs                                                   │
│   patch_tokens (1, P, D) = enc_zoo "dense"                │
│   cls_token    (1, D)    = enc_zoo "cls"  (when it exists)│
└────────────────────────────────────────────────────────────┘
+ <tag>.manifest.json (also embedded in onnx metadata_props):
    schema, tag, hf_id
    input   { height, width, layout: "bgr_hwc_uint8" }
    patch   { size, grid_h, grid_w }        # measured by a real forward
    outputs { patch_tokens: {num, dim}, cls_token: {dim | null} }
    compute_dtype, versions { torch, transformers, vibe sha }
```

why baking is honest and fast:

| concern | reality |
|---|---|
| dtype leak | `uint8→fp32` cast is exact (all 256 values representable); norm ops are the same IEEE elementwise ops torch runs. export-time parity gate (torch vs ORT on random frames) proves it per artifact |
| speed | channel flip / layout are pure data movement, fused by ORT; uint8 H2D upload is 3× smaller than fp32 — baking *wins* |
| dtype policy | training computes tokens in fp16 (`image_feature` default); export `compute_dtype` is a **yaml field** (fp32 default), swappable, recorded in the manifest, measured by the bench |

### the node — zero model math

TCP thread (kept verbatim from v1: reconnecting client, freshness flag) +
timer-gated inference (runs only on a new frame — camera rate decoupled from
tick rate). startup: load manifest → validate every port name/dim against the
graph → prebuild ORT tensors once (zero alloc/tick, cpp_control
`OnnxSession` pattern, CUDA EP fail-loud). per tick: resize → memcpy →
run → publish. spawn banner prints tag, resolution, grid, dim.

### the message — `ImageTokens`

```
std_msgs/Header header      # stamp = frame capture time → staleness gate downstream
uint16 grid_h, grid_w, dim
float32[] cls               # empty for CLS-less backbones (siglip2, mobileclip2)
float32[] patches           # grid_h*grid_w*dim, patch-major raster order
float32 decode_ms, infer_ms # perf stats ride along — no side-channel stats topic
```

`patches` is byte-identical to the onnx `(1, P, D)` tensor — the future
policy node memcpys `msg.patches.data()` straight into its
`kv_tokens__img_tokens` port. topics are model-agnostic: `/enc/tokens`,
`/enc/frame` (post-resize debug frame), QoS keep-last(1) — the policy wants
the latest tokens, never a backlog.

### transport — why vanilla ROS2 suffices

payloads at 112×63 native res, fp32:

| backbone | patch | grid | D | KB/msg | @50 Hz |
|---|---|---|---|---|---|
| theia-tiny | 16 | 7×3 = 21 | 192 | ~17 | 0.8 MB/s |
| dinov2-s / v3-s+ | 14/16 | ≤ 8×4 | 384 | ~50 | 2.5 MB/s |
| tinyclip-39m/16 | 16 | 21 | 512 | ~43 | 2.2 MB/s |
| clip-b32 | 32 | 3×1 = 3 | 768 | ~12 | 0.6 MB/s |
| siglip2-b16 | 16 | 21 | 768 | ~66 | 3.3 MB/s |

worst case 3.3 MB/s on a loopback hop (encoder → policy is same-machine by
design; no wifi in the control loop) — two orders of magnitude below what
CycloneDDS moves. escape hatch if a future backbone goes megabyte-scale:
iceoryx shared memory. fp16-on-wire: not worth the precision seam.

### what got deleted (and why)

| file | verdict |
|---|---|
| `view_front_camera.py` | dead — strict subset of the viewer |
| `fpv_dataset_viewer.py`, `domain_comparison.py`, `tests/implementation_consistency.py` (+ its yaml) | IsaacLab-`fpv_dataset.pt` era; pipeline is mjlab/mocke now. the *workflows* (scrub / A-B / parity) return later against mocke rollouts |
| `recording_node.py` | its only consumer was `domain_comparison` |
| `export_theia_onnx.py` | superseded by the vibe-side zoo exporter (goes with the revamp) |

parity coverage after the deletions: the exporter's built-in torch-vs-ORT
gate per artifact, plus a C++ math surface reduced to `cv::resize` — the
consistency harness has nothing left to check.

## end state

```
vision_encoders/
├── msg/ImageTokens.msg
├── src/encoder_node.cpp            # THE node — zero per-model code
├── config/encoder.yaml             # plumbing only (camera, topics, rate)
├── launch/encoder.launch.py        # model:=<tag> [viewer:=true]
├── scripts/
│   ├── camera_streamer.py          # untouched (best-taste survivor)
│   ├── viewer.py                   # generic — grid/dim from the msg
│   └── wire.py                     # the one python TCP client impl
├── tests/fps.py                    # zoo bench → csv + plot
├── docs/method.md                  # this file
└── models/<tag>.{onnx,manifest.json}   # exported from vibe, tracked there
```
