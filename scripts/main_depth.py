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




# All the connections between keypoints
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

# Variables
edge_color =  (0,0,255) #Red
point_color = (0,255,0) #Green

# Draw the keypoints as circles in the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped_array:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, point_color, -1)
            tf.config.list_physical_devices('GPU')


# Draw the edges between the coordinates
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items(): 
        p1, p2 = edge
        y1, x1, c1 = shaped_array[p1]
        y2, x2, c2 = shaped_array[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), edge_color, 2)


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











if __name__ == '__main__':

	# Loads the model from the file.
	interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
	interpreter.allocate_tensors()
    
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
	
	try:
		with vpi.Backend.CUDA:
			while True:
				
				arr_l = cam_l.image
				arr_r = cam_r.image
				for _ in range(5):
				    _ = cam_l.image
				    _ = cam_r.image
				    time.sleep(0.05)
				
				# RGB -> GRAY
				arr_l = cv2.cvtColor(arr_l, cv2.COLOR_BGR2GRAY)
				arr_r = cv2.cvtColor(arr_r, cv2.COLOR_BGR2GRAY)
				
				# Rectify
				arr_l_rect = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
				arr_r_rect = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)
				
				# Resize
				arr_l_rect = cv2.resize(arr_l_rect, (480, 270))
				arr_r_rect = cv2.resize(arr_r_rect, (480, 270))
				
				# Convert to VPI image
				vpi_l = vpi.asimage(arr_l_rect)
				vpi_r = vpi.asimage(arr_r_rect)
				
				vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
				vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
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
				
				## Apply MoveNet

				# Left
				pose_input_left = cv2.cvtColor(arr_l_rect, cv2.COLOR_GRAY2RGB)
				pose_input_left_resized = tf.image.resize_with_pad(np.expand_dims(pose_input_left, axis=0), 256, 256)
				input_left = tf.cast(pose_input_left_resized, dtype=tf.uint8)

				# Right
				pose_input_right = cv2.cvtColor(arr_r_rect, cv2.COLOR_GRAY2RGB)
				pose_input_right_resized = tf.image.resize_with_pad(np.expand_dims(pose_input_right, axis=0), 256, 256)
				input_right = tf.cast(pose_input_right_resized, dtype=tf.uint8)
				
				# Setup input and output
				input_details = interpreter.get_input_details()
				output_details = interpreter.get_output_details()
				
				# Make predictions for left
				interpreter.set_tensor(input_details[0]['index'], input_left)
				interpreter.invoke()
				keypoints_left = interpreter.get_tensor(output_details[0]['index'])

				# Make predictions for left
				interpreter.set_tensor(input_details[0]['index'], input_right)
				interpreter.invoke()
				keypoints_right = interpreter.get_tensor(output_details[0]['index'])

				# Rendering and showing the image
				draw_img_l = cv2.cvtColor(arr_l_rect, cv2.COLOR_GRAY2BGR)
				draw_img_r = cv2.cvtColor(arr_r_rect, cv2.COLOR_GRAY2BGR)
				
				draw_connections(draw_img_l, keypoints_left, EDGES, 0.4)
				draw_keypoints(draw_img_l, keypoints_left, 0.4)

				draw_connections(draw_img_r, keypoints_right, EDGES, 0.4)
				draw_keypoints(draw_img_r, keypoints_right, 0.4)
				
				# Aynı boyutta değillerse boyutları eşitle
				h = min(draw_img_l.shape[0], draw_img_r.shape[0])
				w = min(draw_img_l.shape[1], draw_img_r.shape[1])
				draw_img_l = cv2.resize(draw_img_l, (w, h))
				draw_img_r = cv2.resize(draw_img_r, (w, h))

				combined = np.hstack((draw_img_l, draw_img_r))

				cv2.imshow("Left + Right Pose", combined)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
		cv2.destroyAllWindows()

