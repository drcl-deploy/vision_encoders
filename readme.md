# vision encoders

Frozen vision backbones as swappable ONNX artifacts: **one C++ node, n models,
zero per-model code.** Sim and real both push frames over TCP
(`<uint32 LE length><JPEG-or-PNG bytes>`); the node decodes, resizes, feeds the
graph (all preprocessing baked in at export), and publishes patch + CLS tokens
on `/enc/tokens` (`msg/ImageTokens`). Design + rationale: [docs/method.md](docs/method.md).

This package only does **inference**. Artifacts are exported from vibe:

```bash
# in the vibe repo / mjlab env — config: src/vibe/export/encoder/export_enc.yaml
export-encoder --tag theia-tiny --out <this_pkg>/models/
export-encoder --out <this_pkg>/models/          # the whole roster
```

Each artifact = `<tag>.onnx` + `<tag>.manifest.json` (vision.onnx.v1 — input
size/layout, patch grid, token dims, CLS presence). The node validates manifest
vs graph at startup and fails loud. `.onnx` files are arch-neutral: export on
x86, run on the Orin.

## install

### thirdparty

Same pattern as [cpp_control](../cpp_control/readme.md#install) — symlink a
pre-built `onnxruntime-linux-<arch>-gpu-<version>` into [thirdparty](./thirdparty/)
and CMake picks the newest. CUDA is required; no CPU fallback.

- **x86_64**: [pre-built release](https://github.com/microsoft/onnxruntime/releases), extract + symlink.
- **aarch64 (Orin)**: build from source (ORT 1.15.x for JetPack 5 / CUDA 11.4,
  1.22+ for JetPack 6 / CUDA 12.x) — recipe in git history or ORT docs.
- CUDA/cuDNN runtime `.so`s can be plucked from the NVIDIA pip wheels into the
  ORT `lib/` dir (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, …) + one
  `patchelf --set-rpath '$ORIGIN' libonnxruntime_providers_cuda.so`.

JetPack 5 note: ORT 1.15 caps at ONNX IR v9 — downgrade the artifact header:
`python3 -c "import onnx; p='models/<tag>.onnx'; m=onnx.load(p); m.ir_version=9; onnx.save(m,p)"`

### build

```bash
colcon build --symlink-install --packages-select vision_encoders
source install/setup.bash
```

## run

Start the streamer first, in its own terminal (sudo needs a tty; the camera
release races the node otherwise). Sim (`unitree_mujoco`) serves the same wire
on `127.0.0.1` — skip the streamer.

```bash
# on the Orin
python3 scripts/camera_streamer.py                 # loopback only
python3 scripts/camera_streamer.py --bind 0.0.0.0  # + offboard viewers
python3 scripts/camera_streamer.py --depth         # + a depth plane (repose planner)

# then, anywhere the workspace is sourced:
ros2 launch vision_encoders encoder.launch.py                    # yaml's model
ros2 launch vision_encoders encoder.launch.py model:=dinov2-s
ros2 launch vision_encoders encoder.launch.py viewer:=true

# offboard viewer (base station, same ROS_DOMAIN_ID):
python3 scripts/viewer.py --camera-ip <orin_wlan0_ip>

# quick check
ros2 topic echo /enc/tokens --no-arr
```

Config ([config/encoder.yaml](config/encoder.yaml)) carries plumbing only —
camera ip/port, raw frame size, topics, tick rate. Everything model-shaped
comes from the manifest.

### the camera wire

`u32 LE length + payload`, one socket. Without `--depth` the payload is JPEG or
PNG bytes and nothing has changed; with it, the payload is a self-describing
multi-plane record — colour and depth in ONE record, so a consumer never pairs
two streams by timestamp. One parser for every consumer:
[include/vision_encoders/frame_wire.hpp](include/vision_encoders/frame_wire.hpp).

```
payload := <JPEG|PNG bytes>                    # legacy, colour only
         | "VD1" u8 nplanes plane[nplanes]
plane   := u8 kind | u8 fmt | u16 w | u16 h | f32 fx,fy,cx,cy | u32 n | bytes
```

Intrinsics are per-plane, in that plane's own pixels, so consumers need no
camera config. Depth ships raw `u16` millimetres at 160×120 (38 KB, no encode —
PNG-16 at full res costs 10–25 ms on the Orin) and `rs.align` runs only on the
frames that carry one (`--depth-hz`). `encoder_node` takes the colour plane and
ignores the rest; depth is [cpp_control's repose planner](../cpp_control/docs/planners/repose/planner.md).
The sim serves the same wire — `camera_depth: 1` in `unitree_mujoco/simulate/config.yaml`.

### supported backbones:

| tag | P (grid) | D | cls | gate | size |
|---|---|---|---|---|---|
| theia-tiny | 21 (3x7) | 192 | y | 6.4e-06 | 22 MB |
| siglip2-b16 | 21 (3x7) | 768 | n | 5.2e-05 | 343 MB |
| dinov2-s | 32 (4x8) | 384 | y | 7.1e-05 | 88 MB |
| dinov3-splus | 21 (3x7) | 384 | y | 1.1e-05 | 115 MB |
| tinyclip-39m | 21 (3x7) | 512 | y | 1.5e-05 | 154 MB |
| clip-b32 | 3 (1x3) | 768 | y | 1.2e-05 | 350 MB |

## topics

| direction | topic | type | notes |
|---|---|---|---|
| pub | `/enc/tokens` | `vision_encoders/ImageTokens` | stamp = capture time; `patches` = (P, D) row-major, memcpy-able into a policy kv port; `cls` empty for CLS-less backbones; decode/infer ms ride along |
| pub | `/enc/frame` | `sensor_msgs/Image` (bgr8) | the model-input frame, as the graph sees it |

## bench

```bash
python3 tests/fps.py                       # every artifact in models/
python3 tests/fps.py --models theia-tiny dinov2-s --duration 20
```

Self-contained (spawns the node + its own frame producer); dumps
`tests/bench/bench_<ts>.{csv,png}` — tokens Hz, decode/infer ms, cpu, rss,
gpu util/mem per model. Measured (RTX, 640x480 wire @ 60 fps, 112x63 input):
all six roster tags hold 60 Hz, infer 4–5.5 ms.

## layout

```
vision_encoders/
├── msg/ImageTokens.msg          # the one token message
├── src/encoder_node.cpp         # THE node — manifest-validated, zero model math
├── config/encoder.yaml          # plumbing only
├── launch/encoder.launch.py     # model:=<tag> [viewer:=true]
├── scripts/
│   ├── camera_streamer.py       # Orin-side TCP producer (real robot)
│   ├── viewer.py                # generic 3-panel viewer (grid/dim from the msg)
│   └── wire.py                  # the one python TCP wire impl
├── tests/fps.py                 # zoo bench
├── docs/method.md               # design + rationale
├── thirdparty/onnxruntime-...   # (symlink)
└── models/<tag>.{onnx,manifest.json}   # export-encoder output (untracked)
```
