import onnxruntime as ort
import numpy as np
import cv2

# Load model
session = ort.InferenceSession("models/movenet-thunder.onnx")

# Get input name
input_name = session.get_inputs()[0].name

# Prepare dummy input image
img = np.ones((1, 256, 256, 3), dtype=np.int32)

# Run inference
outputs = session.run(None, {input_name: img})
keypoints = outputs[0]

print("Keypoints shape:", keypoints[0].shape)
