vision encoders

One node, one wire format. Sim and real both push JPEG frames over TCP
(`<uint32 LE length><JPEG bytes>`); the node decodes, resizes, runs Theia-Tiny
on CUDA, and publishes patch tokens on `/theia_embedding`.

## config

Single source of truth: `config/theia_tiny.yaml`

```yaml
theia_tiny_node:
  ros__parameters:
    hf_name: theaiinstitute/theia-tiny-patch16-224-cdiv
    model_path: models/theia_tiny.onnx
    opset: 17
    cam_width: 640
    cam_height: 480
    camera_ip: "127.0.0.1"   # sim: 127.0.0.1    real: Orin IP (e.g. 192.168.123.164)
    camera_port: 5555
    embedding_topic: /theia_embedding
```

Edit → re-export → rebuild. Export script, launch file, and node all read from it.

## install

### thirdparty

Same pattern as [cpp_control](../cpp_control/readme.md#install) — drop pre-built
libraries into [thirdparty](./thirdparty/) and CMake picks them up. CUDA is
required for this node; there is no CPU fallback.

**1. onnxruntime-gpu v1.22.0**

* download [`onnxruntime-linux-x64-gpu-1.22.0.tgz`](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0)
  and extract it somewhere stable.
* symlink it into `thirdparty/`:

```bash
ln -s /PATH/TO/onnxruntime-linux-x64-gpu-1.22.0 ./vision_encoders/thirdparty/
```

**2. CUDA 12.x + cuDNN 9.x runtime libs**

The ORT prebuilt is linked against CUDA 12.x / cuDNN 9.x. If your system doesn't
already have them (check with `ldconfig -p | grep libcudnn.so.9`), the cleanest
way — no `sudo`, no driver changes, no conflict with conda training envs — is to
pull them from the NVIDIA pip wheels and drop the `.so` files next to the
onnxruntime libs. ORT's `libonnxruntime_providers_cuda.so` has RUNPATH `$ORIGIN`,
so same-dir libs resolve automatically.

```bash
mkdir -p /tmp/cudnn_grab
pip3 install --target=/tmp/cudnn_grab \
  nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

DEST=./vision_encoders/thirdparty/onnxruntime-linux-x64-gpu-1.22.0/lib
cp -av /tmp/cudnn_grab/nvidia/cudnn/lib/libcudnn*.so*         $DEST/
cp -av /tmp/cudnn_grab/nvidia/cublas/lib/libcublas*.so*       $DEST/
cp -av /tmp/cudnn_grab/nvidia/cublas/lib/libcublasLt*.so*     $DEST/
cp -av /tmp/cudnn_grab/nvidia/cuda_runtime/lib/libcudart*.so* $DEST/
cp -av /tmp/cudnn_grab/nvidia/cuda_nvrtc/lib/libnvrtc*.so*    $DEST/

rm -rf /tmp/cudnn_grab
```

The CMake `install()` rule ships every `*.so*` from that dir into
`install/lib/`, so the runtime deps travel with the built node.

Sanity-check before launching:

```bash
ldd install/vision_encoders/lib/libonnxruntime_providers_cuda.so | grep -E 'cudnn|cublas|cudart'
# expect all three resolved — no "not found"
```

(System apt `libcudnn9-cuda-12` / `cuda-runtime-12-6` also works if you prefer
that route; the wheel-pluck above just keeps everything contained inside the
package.)

### export ONNX + build

```bash
pip install torch "transformers==4.41.2" "numpy<2" "Pillow>=9.1" onnx onnxscript onnxruntime einops omegaconf pyyaml
python3 scripts/export_theia_onnx.py

colcon build --symlink-install --packages-select vision_encoders
source install/setup.bash
```

## usage

### real (Orin)

On the Orin, run the camera streamer (bypasses the CycloneDDS 0.7↔0.10 interop bug):

```bash
python3 scripts/camera_streamer.py
```

On the desktop, set `camera_ip` to the Orin's IP and launch:

```bash
ros2 launch vision_encoders theia_tiny.launch.py
```

### sim (unitree_mujoco)

`unitree_mujoco`'s G1 bridge (`unitree_sdk2_bridge.h`) renders the `head_cam`
MJCF camera offscreen and serves identical 640×480 JPEG frames on TCP 5555.
Keep `camera_ip: 127.0.0.1` and launch the node the same way:

```bash
./unitree_mujoco         # in one terminal
ros2 launch vision_encoders theia_tiny.launch.py
```

### check output

```bash
ros2 topic echo /theia_embedding --no-arr
```

### viewer (camera + heatmap)

Opens an OpenCV window showing the exact resized BGR frame going into the
model (top) and the flattened embedding as a heatmap (bottom). The viewer
reads `config/theia_tiny.yaml` and connects as a second TCP client to the
same JPEG stream, so what you see is byte-identical to the node's input.

```bash
# bundled with the node launch
ros2 launch vision_encoders theia_tiny.launch.py viewer:=true

# or standalone (node must already be running somewhere)
python3 cyclonedds_ws/src/vision_encoders/scripts/theia_viewer.py
```

### topics

| direction | topic | type | shape |
|-----------|-------|------|-------|
| pub | `/theia_embedding` | `Float32MultiArray` | (H/16)*(W/16) × 192, layout-encoded |

Images come over raw TCP, not ROS — see `scripts/camera_streamer.py` and
`unitree_sdk2_bridge.h::G1Bridge::renderCamera` for producers.

## dev guide

### changing resolution

1. edit `config/theia_tiny.yaml` → `cam_width` / `cam_height`
2. `python3 scripts/export_theia_onnx.py`
3. `colcon build --symlink-install --packages-select vision_encoders`

Incoming JPEGs are resized in the node to `cam_width × cam_height` before
inference, so the producer resolution doesn't have to match the model — but the
ONNX export must.

### project layout

```
vision_encoders/
├── config/theia_tiny.yaml        # single source of truth
├── launch/theia_tiny.launch.py
├── scripts/
│   ├── camera_streamer.py                      # Orin-side TCP JPEG producer (real)
│   └── export_theia_onnx.py                    # HF model → ONNX
├── src/theia_tiny_node.cpp                     # ROS 2 inference node (CUDA-only)
├── thirdparty/onnxruntime-linux-x64-gpu-1.22.0 # (symlink, see install)
└── models/theia_tiny.onnx                      # (created by export script)
```
