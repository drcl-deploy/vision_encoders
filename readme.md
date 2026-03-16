vision encoders

## config

Single source of truth: `config/theia_tiny.yaml`

```yaml
theia_tiny_node:
  ros__parameters:
    hf_name: theaiinstitute/theia-tiny-patch16-224-cdiv
    model_path: models/theia_tiny.onnx
    opset: 17
    cam_width: 57
    cam_height: 32
    image_topic: /head_camera
    embedding_topic: /theia_embedding
    resolution: 720p
    use_gpu: true
```

Edit this file → re-export → rebuild. The export script, launch file, and node all read from it.

## install

```bash
# 1. download pre-built onnx runtime gpu
bash scripts/setup_onnxruntime.sh

# 2. export model to onnx (reads config/theia_tiny.yaml)
pip install torch "transformers==4.41.2" "numpy<2" "Pillow>=9.1" onnx onnxscript onnxruntime einops omegaconf pyyaml
python3 scripts/export_theia_onnx.py

# 3. build
colcon build --symlink-install --packages-select vision_encoders
source install/setup.bash
```

## usage

```bash
ros2 launch vision_encoders theia_tiny.launch.py
```

check output:

```bash
ros2 topic echo /theia_embedding --no-arr
```

### topics

| direction | topic | type | shape |
|-----------|-------|------|-------|
| sub | `/head_camera` | `Go2FrontVideoData` | raw RGB, cam_width x cam_height x 3 |
| pub | `/theia_embedding` | `Float32MultiArray` | (H//16)*(W//16) x 192, layout-encoded |

output patches = floor(cam_height/16) * floor(cam_width/16).
any resolution works as long as both dimensions are >= 16.

## dev guide

### changing resolution

1. edit `config/theia_tiny.yaml` → `cam_width` / `cam_height`
2. `python3 scripts/export_theia_onnx.py`
3. `colcon build --symlink-install --packages-select vision_encoders`

or one-off override without editing config:

```bash
python3 scripts/export_theia_onnx.py --height 480 --width 640
```

### project layout

```
vision_encoders/
├── config/theia_tiny.yaml         # single source of truth (ROS2 param format)
├── launch/theia_tiny.launch.py    # loads config, passes to node
├── scripts/
│   ├── export_theia_onnx.py       # HF model → ONNX (reads config)
│   └── setup_onnxruntime.sh       # fetch pre-built onnxruntime-gpu
├── src/theia_tiny_node.cpp        # ROS2 inference node
├── thirdparty/onnxruntime-gpu/    # (created by setup script)
└── models/theia_tiny.onnx         # (created by export script)
```
