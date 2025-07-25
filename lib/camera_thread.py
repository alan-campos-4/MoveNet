import cv2
import vpi
import numpy as np
from threading import Thread
from pipeline import gstreamer_pipeline



MAX_DISP = 128
WINDOW_SIZE	= 10


def get_calibration() -> tuple:
	data = np.load("params/disp_params_rectified.npz")
	map_l = (data["map1_l"], data["map2_l"])
	map_r = (data["map1_r"], data["map2_r"])
	return map_l, map_r


# Initialize left and right CSI cameras using GStreamer
class CameraThread(Thread):
	def __init__(self, sensor_id) -> None:
		super().__init__()
		self._cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id), cv2.CAP_GSTREAMER)
		self._should_run = True
		self._image = None
		self.start()
	def run(self):
		while self._should_run:
			ret,frame = self._cap.read()
			if ret:
				self._image = frame
	@property
	def image(self):
		# NOTE: if we care about atomicity of reads, we can add a lock here
		return self._image
	
	def stop(self):
		self._should_run = False
		self._cap.release()
