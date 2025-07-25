import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pose_estimation import *
from pipeline import gstreamer_pipeline
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime

"""		timestamps test 3.py
Basic test to meassure the camera feed framerate of one camera. 
Performs camera feed calibration and pose estimation.
"""



if __name__ == '__main__':

	# Capture variables
	cap = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
	fps = cap.get(cv2.CAP_PROP_FPS)
	font = cv2.FONT_HERSHEY_PLAIN
	
	# Meassurement variables
	timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
	calc_timestamps = [0.0]
	
	# Calibration parameters
	data = np.load("params/stereo_params_undistort.npz")
	K1 = data['K1']
	D1 = data['D1']
	
	# Loads the model from the file.
	interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
	interpreter.allocate_tensors()
	
	while True:
		if (cap.isOpened()==False):
			print("Error: couldn't open the camera.")
			break
		ret, frame = cap.read()
		if not ret:
			print("Error: can't receive frame.")
			break
		
		# Correct distortion
		frame_ud = cv2.undistort(frame, K1, D1)
		
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
		
		# Rendering the pose estimation
		draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
		draw_keypoints(frame, keypoints_with_scores, 0.4)
		
		# Show the feed with framerate
		cv2.putText(frame, str(fps)+' FPS', (20,40), font, 2, (255,255,255), 2, cv2.LINE_AA)
		cv2.imshow('Timestamps', frame)
		
		# Write and compare the framerates
		timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
		calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
		
		# Break the loop if the 'Q' key is pressed
		if cv2.waitKey(10) & 0xFF==ord('q'):
			break
		
	cap.release()
	cv2.destroyAllWindows()
	
	# Displays the results
	for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
		print('Frame %d difference: '%i, abs(ts - cts))


