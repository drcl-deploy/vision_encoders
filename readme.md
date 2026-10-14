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

CUDA is required; there is no CPU fallback. Set up the pinned ONNX Runtime and
build the ROS package with one command:

```bash
./scripts/setup_onnxruntime.sh --build
```

The script detects `x86_64` versus `aarch64`, verifies downloads, installs under
the ignored `.deps/` directory, and maintains the stable
`thirdparty/onnxruntime` link consumed by CMake. It is safe to run repeatedly.

- **x86_64** downloads the checksum-pinned Microsoft ONNX Runtime 1.22 GPU SDK.
- **aarch64 / Orin** detects the CUDA generation and builds the pinned ONNX
  Runtime release from source because Microsoft does not publish an aarch64
  CUDA C++ archive. CUDA 11/cuDNN 8 selects ORT 1.15.1, CUDA 12/cuDNN 8 selects
  ORT 1.18.0, and CUDA 12/cuDNN 9 selects ORT 1.22.0. Other pairs fail early.
- A validated prebuilt Jetson archive can be installed with
  `--archive <path> --sha256 <digest>`.

The script verifies the CUDA provider's shared-library dependencies and fails
before the build if they cannot be resolved. It reuses libraries from the
system, `LD_LIBRARY_PATH`, or installed NVIDIA Python packages. When it finds a
non-system runtime, it installs a ROS environment hook so the C++ node keeps
working after `source install/setup.bash`.

On an x86 workstation with no compatible runtime, install a pinned CUDA
12.8/cuDNN 9 runtime locally (no `sudo`) with:

```bash
./scripts/setup_onnxruntime.sh --install-cuda-runtime --build
```

The download is intentionally opt-in because the NVIDIA runtime is large.
Jetson always uses the CUDA/cuDNN stack supplied by JetPack. The `--build` path
also loads the CUDA provider with `cuda_provider_check`; it does not silently
fall back to CPU. Use `--dry-run` to inspect the ORT selection and `--help` for
all options.

Source builds provision CMake 3.28.4 in a local tool venv and normalize installed
headers to the binary SDK layout. `--jobs 2` (the default) bounds compilation.
JetPack development headers and `nvcc` are required; ORT 1.22 needs GCC 11.1+.
Orin/Xavier architectures are detected from the device tree; override with
`VISION_ENCODERS_CUDA_ARCHITECTURES` when necessary. Inspect a selection without
installing anything with `--dry-run --arch aarch64 --cuda-major 12 --cudnn-major 9`.

JetPack 5 note: the released encoder
artifacts use IR v8/opset 18 and do not require conversion.

Offboard machines can build the unchanged `ImageTokens` interface and headers
without an ONNX/CUDA installation using `-DVISION_ENCODERS_BUILD_RUNTIME=OFF`.
Use separate build/install directories when switching that option. Native ARM
build and inference validation is still required on the target JetPack release.

### build

```bash
colcon build --symlink-install --packages-select vision_encoders \
  --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime  # optional override
source install/setup.bash
ros2 run vision_encoders cuda_provider_check
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
ignores the rest; depth is used by the
[sledgehammer planner](https://github.com/drcl-deploy/sledgehammer/blob/main/docs/planner.md).
The sim serves the same wire — `camera_depth: 1` in `unitree_mujoco/simulate/config.yaml`.

Encoder and planner camera consumers share `camera_tcp.hpp`: connection and
complete-frame reads have a two-second deadline, including partial frames.
EOF or timeout reconnects on a new socket. Shutdown also interrupts connection,
read and retry waits (checked every 100 ms), so a silent camera cannot hold
Ctrl-C indefinitely. Image decoding and inference are unchanged.

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
├── thirdparty/onnxruntime       # stable link managed by setup_onnxruntime.sh
└── models/<tag>.{onnx,manifest.json}   # export-encoder output (untracked)
```
