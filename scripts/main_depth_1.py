import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # enables more tf instructions in operations
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import cv2
import numpy as np
import tensorflow as tf
import vpi
import time
from threading import Thread
from datetime import datetime




"""
	1.	Run MoveNet on the RGB image from both cameras.
	2.	Extract the keypoints.
	3.	Create the disparity map from both cameras.
	4.	Match the coordinates of these keypoints with the corresponding locations in the disparity map.
	5.	Estimate the depth/distance at these keypoints using the disparity values.
"""




""" All the connections between keypoints """
EDGES = {
	(0, 1): 'm',
	(0, 2): 'c',
	(1, 3): 'm',
	(2, 4): 'c',
	(0, 5): 'm',
	(0, 6): 'c',
	(5, 7): 'm',
	(7, 9): 'm',
	(6, 8): 'c',
	(8, 10): 'c',
	(5, 6): 'y',
	(5, 11): 'm',
	(6, 12): 'c',
	(11, 12): 'y',
	(11, 13): 'm',
	(13, 15): 'm',
	(12, 14): 'c',
	(14, 16): 'c',
}

""" Variables """
edge_color =  (0,0,255) #Red
point_color = (0,255,0) #Green


""" Draw the keypoints as circles in the frame """
def draw_keypoints(frame, keypoints, confidence_threshold):
	y, x, c = frame.shape
	shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1]))

	for kp in shaped_array:
		ky, kx, kp_conf = kp
		if kp_conf > confidence_threshold:
			cv2.circle(frame, (int(kx), int(ky)), 4, point_color, -1)


""" Draw the edges between the coordinates """
def draw_connections(frame, keypoints, edges, confidence_threshold):
	y, x, c = frame.shape
	shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1])) # multiplies the keypoints by the dimesions of the frame

	for edge, color in edges.items(): # for every edge, gets the coordinates of the two points and connects them
		p1, p2 = edge
		y1, x1, c1 = shaped_array[p1]
		y2, x2, c2 = shaped_array[p2]

		if (c1 > confidence_threshold) & (c2 > confidence_threshold):
			cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), edge_color, 2)


""" Initialize left and right CSI cameras using GStreamer """
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


""" Get the calibration parameters from the file """
def get_calibration() -> tuple:
	data = np.load("params/disp_params_rectified.npz")
	
	map_l = (data["map1_l"], data["map2_l"])
	map_r = (data["map1_r"], data["map2_r"])
	return map_l, map_r


MAX_DISP = 128
WINDOW_SIZE	= 11






if __name__ == '__main__':

	# Load the model from the file
	interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
	interpreter.allocate_tensors()
	
	# Open both cameras
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
				for _ in range(5):
					_ = cam_l.image
					_ = cam_r.image
					#time.sleep(0.05)
				
				# Rectify the image
				arr_rect_0 = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
				arr_rect_1 = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)
				
				""" 1. Run MoveNet on the RGB image from both cameras. """
				
				# Reshape image
				img0 = arr_rect_0.copy()
				img1 = arr_rect_1.copy()
				img0 = tf.image.resize_with_pad(np.expand_dims(img0, axis=0), 256, 256)
				img1 = tf.image.resize_with_pad(np.expand_dims(img1, axis=0), 256, 256)
				input_image_0 = tf.cast(img0, dtype=tf.float32)
				input_image_1 = tf.cast(img1, dtype=tf.float32)
				
				# Setup input and output
				input_details_0 =	interpreter.get_input_details()
				input_details_1	=	interpreter.get_input_details()
				output_details_0 =	interpreter.get_output_details()
				output_details_1 =	interpreter.get_output_details()
				
				
				""" 2. Extract the keypoints. """
				
				# Make predictions
				interpreter.set_tensor(input_details_0[0]['index'], np.array(input_image_0))
				interpreter.set_tensor(input_details_1[0]['index'], np.array(input_image_1))
				interpreter.invoke()
				keypoints_0 = interpreter.get_tensor(output_details_0[0]['index'])
				keypoints_1 = interpreter.get_tensor(output_details_1[0]['index'])
				
				
				""" 3. Create the disparity map from both cameras. """
				
				# Apply slight Gaussian blur to reduce high-frequency noise
				arr_rect_0 = cv2.GaussianBlur(arr_rect_0, (3, 3), 0)
				arr_rect_1 = cv2.GaussianBlur(arr_rect_1, (3, 3), 0)
				
				# Resize
				arr_rect_0 = cv2.resize(arr_rect_0, (480, 270))
				arr_rect_1 = cv2.resize(arr_rect_1, (480, 270))
				
				# Convert to VPI image
				vpi_l = vpi.asimage(arr_rect_0)
				vpi_r = vpi.asimage(arr_rect_1)
				
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
				disp_arr = disparity_8bpp.cpu()
				disp_arr = cv2.medianBlur(disp_arr, 5)
				disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
				
				
				""" 4. Match the coordinates of these keypoints with the corresponding locations in the disparity map. """
				
				draw_img = disp_arr.copy()
				draw_connections(draw_img,	keypoints_0, EDGES, 0.4)
				draw_keypoints(draw_img,	keypoints_1, 0.4)
				
				
				""" 5. Estimate the depth/distance at these keypoints using the disparity values. """
				
				
				
				

				# Show in one window
				cv2.imshow("Pose Estimation with Depth", draw_img)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
		cv2.destroyAllWindows()












