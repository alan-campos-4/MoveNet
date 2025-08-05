import threading
import tensorflow as tf
import numpy as np
from lib.preprocessing import preprocess

class InferenceWorker(threading.Thread):
	def __init__(self, model, image, device='/GPU:0'):
		super().__init__()
		self.model = model
		self.image = image
		self.device = device
		self.result = None
		self._event = threading.Event()
		
	def run(self):
		print("[Thread] Starting inference")
		input_data = preprocess(self.image)
		with tf.device(self.device):
			outputs = self.model(input=tf.convert_to_tensor(input_data, dtype=tf.int32))
		self.result = outputs["output_0"].numpy()
		print("[Thread] Inference complete")
		self._event.set()


	def get_result(self, timeout=None):
		finished = self._event.wait(timeout)
		if not finished:
			raise TimeoutError("Inference timed out")
		return self.result
