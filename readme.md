vision encoders

One node, one wire format. Sim and real both push frames over TCP
(`<uint32 LE length><JPEG-or-PNG bytes>` — opencv sniffs the magic bytes); the
node decodes, resizes, runs Theia-Tiny on CUDA, and publishes patch tokens on
`/theia_embedding`.

## config

Single source of truth: `config/theia_tiny.yaml`

```yaml
theia_tiny_node:
  ros__parameters:
    hf_name: theaiinstitute/theia-tiny-patch16-224-cddsv
    model_path: models/theia_tiny.onnx        # relative -> resolved against pkg share
    opset: 17

    raw_frame_width: 640                      # wire-format frame size (hardware mirror)
    raw_frame_height: 480
    model_input_width: 57                     # ONNX export size; must match policy training
    model_input_height: 32

    camera_ip: "127.0.0.1"                    # onboard / sim. Offboard: Orin IP (e.g. 192.168.123.164)
    camera_port: 5555

    embedding_topic: /theia_embedding
    inference_rate_hz: 50.0
    publish_input_frame: false                # true -> publish post-resize frame for bag recording
    input_frame_topic: /theia_input_frame
```

Edit → re-export → rebuild. Export script, launch file, and node all read from it.

## install

### thirdparty

Same pattern as [cpp_control](../cpp_control/readme.md#install) — drop pre-built
libraries into [thirdparty](./thirdparty/) and CMake picks them up. CUDA is
required for this node; there is no CPU fallback.

**1. onnxruntime-gpu**

CMake auto-detects architecture and version — just drop a
`onnxruntime-linux-<arch>-gpu-<version>` directory into `thirdparty/`.

**x86_64 (desktop/workstation):** download a pre-built release, e.g.
[`onnxruntime-linux-x64-gpu-1.22.0.tgz`](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0),
extract, and symlink:

```bash
ln -s /PATH/TO/onnxruntime-linux-x64-gpu-1.22.0 ./vision_encoders/thirdparty/
```

**aarch64 (Jetson Orin):** no pre-built GPU release exists — build from source.
Use a version compatible with your JetPack's CUDA (e.g. ORT 1.15.x for CUDA 11.4
on JetPack 5, ORT 1.22+ for CUDA 12.x on JetPack 6):

```bash
git clone --recursive --branch v1.15.1 --depth 1 \
  https://github.com/microsoft/onnxruntime.git
cd onnxruntime
python3 tools/ci_build/build.py \
  --config Release --build_shared_lib --parallel \
  --use_cuda --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --skip_tests --allow_running_as_root --build_dir build \
  --cmake_extra_defines CMAKE_C_COMPILER=gcc-11 CMAKE_CXX_COMPILER=g++-11

# Package into the expected layout
mkdir -p onnxruntime-linux-aarch64-gpu-1.15.1/{lib,include}
cp build/Release/libonnxruntime*.so* onnxruntime-linux-aarch64-gpu-1.15.1/lib/
cp include/onnxruntime/core/session/*.h onnxruntime-linux-aarch64-gpu-1.15.1/include/

ln -s /PATH/TO/onnxruntime-linux-aarch64-gpu-1.15.1 ./vision_encoders/thirdparty/
```

**2. CUDA 12.x + cuDNN 9.x runtime libs**

The ORT prebuilt is linked against CUDA 12.x / cuDNN 9.x. If your system doesn't
already have them (check with `ldconfig -p | grep libcudnn.so.9`), the cleanest
way — no `sudo`, no driver changes, no conflict with conda training envs — is to
pull them from the NVIDIA pip wheels and drop the `.so` files next to the
onnxruntime libs.

```bash
mkdir -p /tmp/cuda_grab
pip3 install --target=/tmp/cuda_grab \
  nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12 nvidia-cufft-cu12

DEST=./vision_encoders/thirdparty/onnxruntime-linux-x64-gpu-<version>/lib
cp -av /tmp/cuda_grab/nvidia/cudnn/lib/libcudnn*.so*         $DEST/
cp -av /tmp/cuda_grab/nvidia/cublas/lib/libcublas*.so*       $DEST/
cp -av /tmp/cuda_grab/nvidia/cublas/lib/libcublasLt*.so*     $DEST/
cp -av /tmp/cuda_grab/nvidia/cuda_runtime/lib/libcudart*.so* $DEST/
cp -av /tmp/cuda_grab/nvidia/cuda_nvrtc/lib/libnvrtc*.so*    $DEST/
cp -av /tmp/cuda_grab/nvidia/curand/lib/libcurand*.so*       $DEST/
cp -av /tmp/cuda_grab/nvidia/cufft/lib/libcufft*.so*         $DEST/

rm -rf /tmp/cuda_grab
```

The ORT provider `.so` ships without RUNPATH, so patch it once after extracting:

```bash
pip3 install patchelf   # if not already installed
patchelf --set-rpath '$ORIGIN' $DEST/libonnxruntime_providers_cuda.so
```

The CMake `install()` rule ships every `*.so*` from that dir into
`install/lib/`, and the node binary's RPATH points back to that dir, so
the runtime deps travel with the built node.

Sanity-check before launching:

```bash
ldd install/vision_encoders/lib/libonnxruntime_providers_cuda.so \
  | grep -E 'cudnn|cublas|cudart|curand|cufft'
# expect all resolved — no "not found"
```

(System apt `libcudnn9-cuda-12` / `cuda-runtime-12-6` etc. also works if you
prefer that route; the wheel-pluck above just keeps everything contained inside
the package.)

### export ONNX + build

```bash
pip install torch "transformers>=4.42,<4.46" "huggingface-hub<0.30" \
  "numpy<2" "Pillow>=9.1" onnx onnxscript onnxruntime einops omegaconf pyyaml
python3 scripts/export_theia_onnx.py

colcon build --symlink-install --packages-select vision_encoders
source install/setup.bash
```

Notes on the pins:
- `transformers>=4.42` — older versions hit a `nn.functional.interpolate(scale_factor=…)`
  TypeError during `torch.onnx.export` tracing under torch ≥ 2.3 (vit's
  `interpolate_pos_encoding`). 4.42+ uses `size=(int, int)` which traces cleanly.
- `huggingface-hub<0.30` — newer versions pull `hf-xet`, which has no wheel for
  Python 3.8 / aarch64 (JetPack 5).
- Theia's HF code uses `match`/`case`, so the export needs Python ≥ 3.10. On
  JetPack 5 the system Python is 3.8 — use `python3.10` explicitly, or run
  export on a dev machine and `scp` the `.onnx` over.

#### IR-version downgrade (only when ORT < model IR)

Newer torch (≥ 2.4) exports ONNX with IR v10. The Jetson Orin's onnxruntime
1.15.x (built for JetPack 5 / CUDA 11.4) caps at IR v9, so loading the freshly
exported model fails with `Unsupported model IR version: 10`. Downgrade the
IR field in place — opset-17 ops are all valid under IR v9, so this is purely
a header rewrite:

```bash
python3 -c "import onnx; p='models/theia_tiny.onnx'; m=onnx.load(p); m.ir_version=9; onnx.save(m,p)"
```

Skip this on x86_64 setups using a recent ORT (≥ 1.16 handles IR v10 natively).

## launch

Start the streamer first in its own terminal — it's intentionally decoupled
from the launch (sudo needs a tty for `mscli stopservice`, and the camera
release races the node otherwise):

- **Onboard (default)**: `python3 scripts/camera_streamer.py` on the Orin.
- **Sim**: `unitree_mujoco` already serves the same stream on `127.0.0.1` — skip the streamer.
- **Offboard dev**: `python3 scripts/camera_streamer.py --bind 0.0.0.0`, then set `camera_ip` in the yaml to the Orin's IP.

Then:

```bash
# inference only
ros2 launch vision_encoders theia_tiny.launch.py

# + OpenCV viewer (camera frame + embedding heatmap)
ros2 launch vision_encoders theia_tiny.launch.py viewer:=true

# + joystick-gated bag recorder (A starts, Y/R1 stops)
# bag lands at <dir>/<YYYYMonDD_HHMM>/
ros2 launch vision_encoders theia_tiny.launch.py record:=/path/to/scenario_dir
```

quick check:

```bash
ros2 topic echo /theia_embedding --no-arr
```

## individual scripts

```bash
# Orin-side TCP frame producer (defaults: JPEG@85, bind 127.0.0.1)
# flags: --codec {jpeg,png}  --bind <addr>  -v / --verbose
python3 scripts/camera_streamer.py

# export HuggingFace Theia -> ONNX (reads config/theia_tiny.yaml)
python3 scripts/export_theia_onnx.py

# standalone live viewer (node must be running)
python3 scripts/theia_viewer.py

# scrub an IsaacLab fpv_dataset.pt interactively
python3 scripts/fpv_dataset_viewer.py --dataset /path/to/fpv_dataset_dir

# A/B compare IsaacLab dataset vs recorded rollout
python3 scripts/domain_comparison.py --source /path/to/fpv_dataset_dir --target /path/to/bag_dir

# C++ <-> IsaacLab encoder parity test (MSE / cosine)
python3 tests/implementation_consistency.py --dataset /path/to/fpv_dataset_dir
```

## topics

| direction | topic | type | shape |
|-----------|-------|------|-------|
| pub | `/theia_embedding`     | `Float32MultiArray` | (H/16)*(W/16) × 192, layout-encoded |
| pub | `/theia_input_frame`   | `sensor_msgs/Image` (rgb8) | post-resize, pre-norm; only when recording |

Images come over raw TCP, not ROS — see `scripts/camera_streamer.py` and
`unitree_sdk2_bridge.h::G1Bridge::renderCamera` for producers.

## dev guide

### changing resolution

The node does two resizes: wire frame → `raw_frame_*` (sanity-mirror of the
hardware output) → `model_input_*` (the size the ONNX was exported at and the
policy was trained against).

1. edit `config/theia_tiny.yaml` → `model_input_width` / `model_input_height`
2. `python3 scripts/export_theia_onnx.py`  &nbsp;# re-exports ONNX at the new size
3. `colcon build --symlink-install --packages-select vision_encoders`

`raw_frame_*` rarely changes — it mirrors the D435i's native output.

### project layout

```
vision_encoders/
├── config/
│   ├── theia_tiny.yaml                         # single source of truth
│   └── theia_tiny_consistency_test.yaml        # parity-test override (no-op resizes)
├── launch/theia_tiny.launch.py
├── scripts/
│   ├── camera_streamer.py                      # Orin-side TCP JPEG producer (real)
│   ├── export_theia_onnx.py                    # HF model -> ONNX
│   ├── theia_viewer.py                         # live camera + heatmap
│   ├── fpv_dataset_viewer.py                   # scrub IsaacLab fpv dataset
│   ├── domain_comparison.py                    # sim vs rollout side-by-side
│   └── recording_node.py                       # joystick-gated bag writer
├── tests/implementation_consistency.py         # C++/IsaacLab encoder parity
├── src/theia_tiny_node.cpp                     # ROS 2 inference node (CUDA-only)
├── thirdparty/onnxruntime-linux-<arch>-gpu-<ver> # (symlink, see install)
└── models/theia_tiny.onnx                      # (created by export script)
```
