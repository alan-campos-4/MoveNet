import onnx
from onnx import TensorProto

# Load the existing model
model_path = "models/movenet-thunder.onnx"
model = onnx.load(model_path)

# Set the input data type to float32
model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT

# Save the new model
onnx.save(model, "models/movenet-thunder-f32.onnx")
print("New model saved: movenet-thunder-f32.onnx")
