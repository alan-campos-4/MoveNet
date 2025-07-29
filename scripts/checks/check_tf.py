import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	print("Num GPUs Available: ", len(gpus))
	for var in model.variables:
		print(f"{var.name} is placed on {var.device}")
else:
	print("No GPUs available.")
