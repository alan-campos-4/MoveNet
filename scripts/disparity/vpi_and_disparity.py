import cv2
import numpy as np
import vpi
import time
from threading import Thread
from datetime import datetime
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline_calibration import gstreamer_pipeline


MAX_DISP = 128
WINDOW_SIZE	= 10


def get_calibration() -> tuple:
	data = np.load("params/stereo_params_undistort.npz")
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
	
	print("Left frame is None?", frame_l is None, "Right frame is None?", frame_r is None)
	if frame_l is not None:
		cv2.imshow("Test Left", frame_l)
	if frame_r is not None:
		cv2.imshow("Test Right", frame_r)
	cv2.waitKey(1000)
	cv2.destroyAllWindows()
	
	try:
		with vpi.Backend.CUDA:
			for i in range(100):
				ts = []
				ts.append(time.perf_counter())
				
				arr_l = cam_l.image
				arr_r = cam_r.image
				for _ in range(5):
				    _ = cam_l.image
				    _ = cam_r.image
				    time.sleep(0.05)
				ts.append(time.perf_counter())
				
				# RGB -> GRAY
				arr_l = cv2.cvtColor(arr_l, cv2.COLOR_BGR2GRAY)
				arr_r = cv2.cvtColor(arr_r, cv2.COLOR_BGR2GRAY)
				ts.append(time.perf_counter())
				
				# Rectify
				arr_l_rect = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
				arr_r_rect = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)
				ts.append(time.perf_counter())
				
				# Resize
				arr_l_rect = cv2.resize(arr_l_rect, (480, 270))
				arr_r_rect = cv2.resize(arr_r_rect, (480, 270))
				ts.append(time.perf_counter())
				
				# Convert to VPI image
				vpi_l = vpi.asimage(arr_l_rect)
				vpi_r = vpi.asimage(arr_r_rect)
			
				vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
				vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
			
				vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
				vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
				ts.append(time.perf_counter())
				
				disparity_16bpp = vpi.stereodisp(
					vpi_l_16bpp,
					vpi_r_16bpp,
					out_confmap = None,
					backend = vpi.Backend.CUDA,
					window = WINDOW_SIZE,
					maxdisp = MAX_DISP,
				)
				disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32 * MAX_DISP))
				ts.append(time.perf_counter())
			
				disp_arr = disparity_8bpp.cpu()
				ts.append(time.perf_counter())
			
				disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
				ts.append(time.perf_counter())
			
				cv2.imshow("Disparity", disp_arr)
				cv2.waitKey(1)
				ts.append(time.perf_counter())
			
				ts = np.array(ts)
				ts_deltas = np.diff(ts)
			
				debug_str = f"Iter {i}\n"
			
				for task, dt in zip(
					[
						"Read images",
						"OpenCV RGB->GRAY",
						"OpenCV Rectify",
						"OpenCV 1080p->270p Resize",
						"VPI conversions",
						"Disparity calc",
						".cpu() mapping",
						"OpenCV colormap",
						"Render",
					],
					ts_deltas,
				):
					debug_str += f"{task} {1000*dt:0.2f}\n"
				
				print(debug_str)
				
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
