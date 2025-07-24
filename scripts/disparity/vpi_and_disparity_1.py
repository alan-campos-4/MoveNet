import cv2
import numpy as np
import vpi
import time
from threading import Thread
from datetime import datetime
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline


MAX_DISP = 128
WINDOW_SIZE	= 11


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


if __name__ == "__main__":

	map_l, map_r = get_calibration()
	cam_l = CameraThread(0)
	cam_r = CameraThread(1)
	
	time.sleep(0.5)
	for _ in range(5):
		_ = cam_l.image
		_ = cam_r.image
		time.sleep(0.05)
	frame_l = cam_l.image
	frame_r = cam_r.image

	try:
		with vpi.Backend.CUDA:
			while True:
				arr_l = cam_l.image
				arr_r = cam_r.image
				
				# RGB -> GRAY
				arr_l = cv2.cvtColor(arr_l, cv2.COLOR_BGR2GRAY)
				arr_r = cv2.cvtColor(arr_r, cv2.COLOR_BGR2GRAY)
				
				# Rectify
				arr_l_rect = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
				arr_r_rect = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)
				
				# Apply slight Gaussian blur to reduce high-frequency noise
				arr_l_rect = cv2.GaussianBlur(arr_l_rect, (3, 3), 0)
				arr_r_rect = cv2.GaussianBlur(arr_r_rect, (3, 3), 0)
				
				# Resize
				arr_l_rect = cv2.resize(arr_l_rect, (480, 270))
				arr_r_rect = cv2.resize(arr_r_rect, (480, 270))
				
				# Convert to VPI image
				vpi_l = vpi.asimage(arr_l_rect)
				vpi_r = vpi.asimage(arr_r_rect)
				vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
				vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
				
				disparity_16bpp = vpi.stereodisp(
					vpi_l_16bpp,
					vpi_r_16bpp,
					out_confmap = None,
					backend = vpi.Backend.CUDA,
					window = WINDOW_SIZE,
					maxdisp = MAX_DISP,
				)
				disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32 * MAX_DISP))
				disp_arr = disparity_8bpp.cpu()
				disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
				
				cv2.imshow("Disparity", disp_arr)
				
				if cv2.waitKey(1) & 0xFF==ord('q'):
					break
				
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
