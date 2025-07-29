import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
from pose_estimation import *
from timestamps import *
import numpy as np
import cv2
import time
import tensorflow as tf


"""
Meassures the framerate of a camera while performing calibration and pose estimation.
"""



if __name__ == '__main__':
	
	# TensorFlow logging
	tf.debugging.set_log_device_placement(True)

	# Opens the camera
	cap = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
	if (cap.isOpened()==False):
		print("Error: couldn't open the camera.")
		exit()
	
	# Meassurement variables
	start_time = time.time()
	frame_count = 0
	fps = 0
	fps_array = []
	seconds_passed = 0
	max_seconds = get_max_seconds()
	
	# Calibration parameters
	data = np.load("params/stereo_params_undistort.npz")
	K1 = data['K1']
	D1 = data['D1']
	
	
	
	# Restrict TensorFlow to only allocate a specific amount of GPU memory if necessary
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		try:
			# Set memory growth option
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError as e:
			print(e)
	
	# Check GPU details
	print("TensorFlow version:", tf.__version__)
	print("Available GPUs: ", gpus)
	
	# Loads GPU device
	with tf.device('/GPU:0'):
	
		# Loads the model from the file.
		interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
		interpreter.allocate_tensors()
		
		try:
			while seconds_passed < max_seconds:		
				ret, frame = cap.read()
				if not ret:
					print("Error: can't receive frame.")
					break
				
				# Calibrates the camera
				frame = cv2.undistort(frame, K1, D1)
				
				# Adds the time passed
				frame_count += 1
				elapsed_time = time.time()-start_time
				# Calculates the frames per second in the time passed
				if elapsed_time > 1.0:
					seconds_passed += 1
					fps = frame_count / elapsed_time
					fps_array.append(fps)
					frame_count = 0
					start_time = time.time()
				
				# Reshape image
				img = frame.copy()
				img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
				input_image = tf.cast(img, dtype=tf.float32)
				
				# Setup input and output
				input_details = interpreter.get_input_details()
				output_details = interpreter.get_output_details()
				
				# Make predictions
				interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
				interpreter.invoke()
				keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
				
				# Rendering and showing the image
				draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
				draw_keypoints(frame, keypoints_with_scores, 0.4)
				
				# Shows the feed with framerate
				show_text(cv2, frame, seconds_passed, max_seconds, fps)
				cv2.imshow('Timestamps', frame)
				
				# Break the loop if the 'Q' key is pressed
				if cv2.waitKey(10) & 0xFF==ord('q'):
					break
				
			# Saves the results
			save_performance(__file__, 'Calibration and pose estimation', fps_array, max_seconds, cap)
			
		except KeyboardInterrupt as e:
			print(e)
		finally:
			cap.release()
			cv2.destroyAllWindows()

