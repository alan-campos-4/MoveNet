🚀 Deploying MoveNet on Jetson Xavier NX using ONNX + TensorRT

📌 Overview

This guide explains how to deploy Google’s MoveNet (Thunder) pose estimation model on Jetson Xavier NX using ONNX Runtime with TensorRT acceleration. This approach ensures high performance inference on Jetson’s GPU.

We will use a pre-optimized ONNX model provided by PINTO0309, convert it to a TensorRT engine (optional), and perform real-time inference using onnxruntime.


📁 Step 1 – Clone the MoveNet ONNX repository

We will use the optimized MoveNet ONNX model by PINTO0309.
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/130_MoveNet/model

Download the pre-converted ONNX model:
wget https://github.com/PINTO0309/PINTO_model_zoo/raw/main/130_MoveNet/model/movenet_thunder_256x256.onnx


⚙️ Step 2 – Convert ONNX to TensorRT engine (optional)

You can convert the ONNX model to a .trt engine for faster cold start and even better GPU optimization. This step is optional—onnxruntime can also use ONNX directly.
trtexec --onnx=movenet_thunder_256x256.onnx --saveEngine=movenet_thunder.trt --fp16

--fp16: Enables half-precision mode for better performance.
Requires JetPack 4.6+ or 5.x.


🧪 Step 3 – Run Inference with ONNX Runtime + TensorRT

Make sure onnxruntime-gpu is installed:
pip install onnxruntime-gpu

Then, create a Python script to run inference:
import onnxruntime as ort
import numpy as np

# Load the ONNX model using TensorRT execution provider
session = ort.InferenceSession(
    "movenet_thunder_256x256.onnx",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"]
)

# Prepare dummy input for test (replace with actual preprocessed image)
input_name = session.get_inputs()[0].name
input_data = np.random.rand(1, 256, 256, 3).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})

# Print model output (keypoints)
print("Keypoints:", outputs[0])



Tip: During runtime, monitor GPU usage using:
sudo tegrastats

If GR3D_FREQ rises during inference, your model is using the GPU.
