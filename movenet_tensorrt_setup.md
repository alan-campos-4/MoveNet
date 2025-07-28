# ========================================================
# 1. Download MoveNet Thunder model (TensorFlow SavedModel)
# ========================================================
mkdir -p movenet/thunder
wget https://tfhub.dev/google/movenet/singlepose/thunder/4?tf-hub-format=compressed -O thunder.tar.gz
tar -xvf thunder.tar.gz -C movenet/thunder

# After extraction, you should have:
# movenet/thunder/saved_model.pb
# movenet/thunder/variables/

# ========================================================
# 2. Install tf2onnx to convert the model to ONNX format
# ========================================================
pip install tf2onnx

# ========================================================
# 3. Convert the SavedModel to ONNX
# ========================================================
python -m tf2onnx.convert \
    --saved-model movenet/thunder \
    --output movenet_thunder.onnx \
    --opset 13

# ========================================================
# 4. (Optional) Verify the ONNX model
# ========================================================
# This requires onnx and netron libraries
pip install onnx netron
netron movenet_thunder.onnx  # Opens model structure in browser

# ========================================================
# 5. Convert ONNX model to TensorRT engine
# ========================================================
/usr/src/tensorrt/bin/trtexec \
    --onnx=movenet_thunder.onnx \
    --saveEngine=movenet_thunder.trt \
    --fp16

# Note: --fp16 flag enables FP16 precision (faster inference if supported)

# ========================================================
# 6. Install ONNX Runtime with GPU support
# ========================================================
pip install onnxruntime-gpu

# ========================================================
# 7. Python Inference Example using TensorRT Engine
# ========================================================
# Save this as run_movenet_inference.py or similar

import onnxruntime as ort
import numpy as np
import cv2

# Load session with GPU and TensorRT providers
session = ort.InferenceSession("movenet_thunder.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

# Load and prepare input image
img = cv2.imread("sample.jpg")  # or use camera input
img = cv2.resize(img, (256, 256))
input_image = img.astype(np.float32)[np.newaxis, ...]  # Shape: (1, 256, 256, 3)

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_image})

# Parse results
keypoints = output[0][0][0]  # shape: (17, 3)
for idx, (x, y, confidence) in enumerate(keypoints):
    if confidence > 0.2:
        print(f"Keypoint {idx}: x={x:.2f}, y={y:.2f}, conf={confidence:.2f}")
