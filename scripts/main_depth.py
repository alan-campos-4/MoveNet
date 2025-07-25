import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pose_estimation import *
from pipeline import gstreamer_pipeline
import cv2
import numpy as np
import tensorflow as tf
import vpi
import time
from threading import Thread
from datetime import datetime




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


def get_calibration() -> tuple:
	data = np.load("params/disp_params_rectified.npz")
	map_l = (data["map1_l"], data["map2_l"])
	map_r = (data["map1_r"], data["map2_r"])
	return map_l, map_r


MAX_DISP = 128
WINDOW_SIZE	= 10





if __name__ == '__main__':
	
	# Loads the model from the file.
	interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
	interpreter.allocate_tensors()
	
	map_l, map_r = get_calibration()
	cam_l = CameraThread(0)
	cam_r = CameraThread(1)
	
	#time.sleep(0.5)
	#for _ in range(5):
		#_ = cam_l.image
		#_ = cam_r.image
		#time.sleep(0.05)
	#frame_l = cam_l.image
	#frame_r = cam_r.image

	print("Waiting for the first valid frames from the cameras...")
	while cam_l.image is None or cam_r.image is None:
    		time.sleep(0.01)
	print("Frames received. Starting the application.")
	
	try:
		with vpi.Backend.CUDA:
			while True:
				arr_l = cam_l.image
				arr_r = cam_r.image
				#for _ in range(5):
					#_ = cam_l.image
					#_ = cam_r.image
					#time.sleep(0.05)
				
				# RGB -> GRAY
				#arr_l = cv2.cvtColor(arr_l, cv2.COLOR_BGR2GRAY)
				#arr_r = cv2.cvtColor(arr_r, cv2.COLOR_BGR2GRAY)
				
				# Rectify
				arr_l_rect = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
				arr_r_rect = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)

				# Reshape image
				img_l = arr_l_rect.copy()
				img_r = arr_r_rect.copy()
				input_image_l = tf.image.resize_with_pad(np.expand_dims(img_l, axis=0), 256, 256)
				input_image_r = tf.image.resize_with_pad(np.expand_dims(img_r, axis=0), 256, 256)
				input_image_l = tf.cast(input_image_l, dtype=tf.float32) / 255.0
				input_image_r = tf.cast(input_image_r, dtype=tf.float32) / 255.0

				# Setup input and output
				input_details_l = interpreter.get_input_details()
				input_details_r = interpreter.get_input_details()
				output_details_l = interpreter.get_output_details()
				output_details_r = interpreter.get_output_details()
	
				# Make predictions
				interpreter.set_tensor(input_details_l[0]['index'], np.array(input_image_l))
				interpreter.set_tensor(input_details_r[0]['index'], np.array(input_image_r))
				interpreter.invoke()
				keypoints_l = interpreter.get_tensor(output_details_l[0]['index'])
				keypoints_r = interpreter.get_tensor(output_details_r[0]['index'])
				
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
				
				disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32*MAX_DISP) )
				disp_raw = disparity_16bpp.convert(vpi.Format.U8, scale=1.0).cpu()
				disp_arr = disparity_8bpp.cpu()
				disp_arr = cv2.medianBlur(disp_arr, 5)
				disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
				
				# Depth estimation parameters
				baseline = 0.1		# Distance between cameras in meters
				focal_length = 580	# Focal length in pixels (from calibration)

				cx = 240
				cy = 135
				
				draw_img = disp_arr.copy()
				
				# Image size for keypoint mapping
				h, w = draw_img.shape[:2]
				
				# Iterate over each keypoint to calculate 3D position using disparity
				# (Only print elbows for performance)
				for idx, (y_norm, x_norm, conf) in enumerate(keypoints_l[0][0]):
					if conf < 0.4:
						continue
					x = int(x_norm * w)
					y = int(y_norm * h)
					if 0 <= x < w and 0 <= y < h:
						# Get disparity value at keypoint location
						disparity_val = disp_raw[y, x] / 32.0
						if disparity_val > 0:
							# Calculate depth (z), and 3D coordinates (X, Y)
							z = (baseline * focal_length) / disparity_val
							X = (x - cx) * z / focal_length
							Y = (y - cy) * z / focal_length
							
							label = f"x:{X:.2f}m, y:{Y:.2f}m, z:{z:.2f}m"
							cv2.circle(draw_img, (x, y), 4, (0, 255, 0), -1)
							cv2.putText(draw_img, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
				
							# Print only elbow keypoints (left = 7, right = 8)
							if idx == 7:
								print(f"[Left Elbow] x: {X:.2f} m, y: {Y:.2f} m, z: {z:.2f} m")
							elif idx == 8:
								print(f"[Right Elbow] x: {X:.2f} m, y: {Y:.2f} m, z: {z:.2f} m")
				
				# Draw skeleton on frame
				draw_connections(draw_img, keypoints_l, EDGES, 0.4)
				draw_keypoints(draw_img, keypoints_r, 0.4)
				
				# Show both disparity and pose estimation results
				cv2.imshow("Pose Estimation with Depth", draw_img)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
		cv2.destroyAllWindows()

