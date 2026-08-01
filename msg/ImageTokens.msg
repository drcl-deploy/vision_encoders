# One frozen-encoder forward: patch + CLS tokens, layout mirroring the onnx
# (1, P, D) output — `patches` is patch-major raster order (row grid_h, col
# grid_w), memcpy-able straight into a policy graph's kv-token port.

std_msgs/Header header    # stamp = frame capture time -> staleness gating downstream

uint16 grid_h
uint16 grid_w
uint16 dim

float32[] cls             # dim floats; empty when the backbone has no CLS token
float32[] patches         # grid_h * grid_w * dim floats

# perf stats ride along (no side-channel topic)
float32 decode_ms         # tcp recv -> imdecode -> stage-1 resize
float32 infer_ms          # stage-2 resize -> onnx -> publish
