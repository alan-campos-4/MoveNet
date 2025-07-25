import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # enables more tf instructions in operations
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pose_estimation import *
from pipeline import gstreamer_pipeline
import cv2
import numpy as np
import tensorflow as tf








if __name__ == '__main__':

	# Loads the model from the file.
	interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
	interpreter.allocate_tensors()

	# Reads the camera and captures the video.
	cap = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

	while True:
		if (cap.isOpened()==False):
			print("Error: couldn't open the camera.")
			break

		ret, frame = cap.read()

		if not ret:
			print("Error: can't receive frame.")
			break

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

		# Display the result 
		cv2.imshow('Movenet Thunder', frame)

		# Break the loop if the 'Q' key is pressed
		if cv2.waitKey(10) & 0xFF==ord('q'):
			break


	cap.release()
	cv2.destroyAllWindows()
