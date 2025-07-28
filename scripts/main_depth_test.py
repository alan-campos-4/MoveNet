import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # enables more tf instructions in operations
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
from camera_thread import *
from pose_estimation import *
import cv2
import numpy as np
import tensorflow as tf
import vpi
import time



"""
	1.	Run MoveNet on the RGB image from both cameras.
	2.	Extract the keypoints.
	3.	Create the disparity map from both cameras.
	4.	Match the coordinates of these keypoints with the corresponding locations in the disparity map.
	5.	Estimate the depth/distance at these keypoints using the disparity values.
"""



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
				#data = np.load("params/disp_params_rectified.npz")
				#K = data["K1"]
				#print("Focal length (fx):", K[0, 0])
		  
				#T = data["T"]
				#baseline_m = abs(T[0])
				#baseline_cm = baseline_m * 100
				#print("Baseline:", baseline_cm, "cm")
		  
				focal_length = 752.90670806571
				baseline_cm = 7.74058794
				
				keypoints = keypoints_0[0][0]
				
				disp_raw = disparity_16bpp.cpu().view(np.ndarray)
				
				for i, (x, y, conf) in enumerate(keypoints):
					if conf > 0.2:
						x_disp = int(x * disp_raw.shape[1])
						y_disp = int(y * disp_raw.shape[0])
						if 0 <= x_disp < disp_raw.shape[1] and 0 <= y_disp < disp_raw.shape[0]:
							disparity_val = disp_raw[y_disp, x_disp]
							if disparity_val > 0:
								real_disparity = disparity_val / 32.0
								Z = (focal_length * baseline_cm) / real_disparity
								print(f"Keypoint {i} depth: {Z:.1f} cm")
								cv2.circle(draw_img, (x_disp, y_disp), 4, (0, 255, 255), -1)
								cv2.putText(draw_img, f"{int(Z)} cm", (x_disp + 5, y_disp - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				
				# Show in one window
				cv2.imshow("Depth-annotated keypoints", draw_img)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
		cv2.destroyAllWindows()
