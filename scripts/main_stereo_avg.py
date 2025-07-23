import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # enables more tf instructions in operations
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import cv2
import numpy as np
import tensorflow as tf




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
color_red = (0,0,255) #Red
color_grn = (0,255,0) #Green
color_blu = (255,0,0) #Blue

# Draw the keypoints as circles in the frame
def draw_keypoints(frame, keypoints, confidence_threshold, color=(0,255,0)):
	y, x, c = frame.shape
	shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1]))
	for kp in shaped_array:
		ky, kx, kp_conf = kp
		if kp_conf > confidence_threshold:
			cv2.circle(frame, (int(kx), int(ky)), 4, color, -1)

# Draw the edges between the coordinates
def draw_connections(frame, keypoints, edges, confidence_threshold, colorN=(0,0,255)):
	y, x, c = frame.shape
	shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1])) # multiplies the keypoints by the dimesions of the frame
	for edge, color in edges.items(): # for every edge, gets the coordinates of the two points and connects them
		i1, i2 = edge
		y1, x1, c1 = shaped_array[i1]
		y2, x2, c2 = shaped_array[i2]
		if (c1 > confidence_threshold) & (c2 > confidence_threshold):
			cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), colorN, 2)






if __name__ == '__main__':

	# Loads the model from the file.
	interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
	interpreter.allocate_tensors()
	
	data = np.load("params/stereo_params_undistort.npz")
	K1 = data['K1']  # Camera 1 intrinsic matrix
	D1 = data['D1']  # Camera 1 distortion coefficients
	K2 = data['K2']  # Camera 2 intrinsic matrix
	D2 = data['D2']  # Camera 2 distortion coefficients
	R =  data['R']   # Rotation matrix between cameras
	T =  data['T']   # Translation vector between cameras

	# Open both cameras
	width = 640
	height = 360
	cap0 = cv2.VideoCapture(gstreamer_pipeline(0, width, height), cv2.CAP_GSTREAMER)
	cap1 = cv2.VideoCapture(gstreamer_pipeline(1, width, height), cv2.CAP_GSTREAMER)

	if not cap0.isOpened():
		print("Error: Could not open camera 0")
		exit()
	if not cap1.isOpened():
		print("Error: Could not open camera 1")
		exit()

	while True:
		ret0, frame0 = cap0.read()
		ret1, frame1 = cap1.read()
		
		if not ret0 or not ret1:
			print("Error: Could not read from one or both cameras.")
			break
			
		frame0_ud = cv2.undistort(frame0, K1, D1)
		frame1_ud = cv2.undistort(frame1, K2, D2)
		
		# Reshape image
		img0 = frame0_ud.copy()
		img1 = frame1_ud.copy()
		img0 = tf.image.resize_with_pad(np.expand_dims(img0, axis=0), 256, 256)
		img1 = tf.image.resize_with_pad(np.expand_dims(img1, axis=0), 256, 256)
		input_image_0 = tf.cast(img0, dtype=tf.float32)
		input_image_1 = tf.cast(img1, dtype=tf.float32)
		
		# Setup input and output
		input_details_0 = interpreter.get_input_details()
		input_details_1 = interpreter.get_input_details()
		output_details_0 = interpreter.get_output_details()
		output_details_1 = interpreter.get_output_details()
		
		# Make predictions
		interpreter.set_tensor(input_details_0[0]['index'], np.array(input_image_0))
		interpreter.set_tensor(input_details_1[0]['index'], np.array(input_image_1))
		interpreter.invoke()
		keypoints_with_scores_0 = interpreter.get_tensor(output_details_0[0]['index'])
		keypoints_with_scores_1 = interpreter.get_tensor(output_details_1[0]['index'])
		
		#Average of the keypoints
		keypoints_avg = []
		
		for block0, block1 in zip(keypoints_with_scores_0, keypoints_with_scores_1):
			new_block = []
			for col0, col1 in zip(block0, block1):
				new_col = []
				for row0, row1 in zip(col0, col1):
					new_row = []
					for num0, num1 in zip(row0, row1):
						new_num = (num0 + num1)/2
						new_row.append(new_num)
					new_col.append(new_row)
				new_block.append(new_col)
			keypoints_avg.append(new_block)
		
		print(keypoints_with_scores_0)
		print(np.array(keypoints_avg))
		
		# Create blank frames for the estimation comparison
		frame_avg =	 np.zeros((height,width,3), np.uint8)
		frame_left = np.zeros((height,width,3), np.uint8)
		
		# Rendering and showing the image
		draw_connections(frame_avg, keypoints_avg, EDGES, 0.4, (255,0,0) )
		draw_keypoints(frame_avg,   keypoints_avg, 0.4)
		draw_connections(frame_left, keypoints_with_scores_0, EDGES, 0.4)
		draw_keypoints(frame_left,   keypoints_with_scores_0, 0.4)
		draw_connections(frame_left, keypoints_with_scores_1, EDGES, 0.4, (255,255,0))
		draw_keypoints(frame_left,   keypoints_with_scores_1, 0.4, (255,255,255))
		
		# Resize
		#frame0_ud	= cv2.resize(frame0_ud, (480, 270))
		#frame1_ud	= cv2.resize(frame1_ud, (480, 270))
		frame_avg	= cv2.resize(frame_avg, (480, 270))
		frame_left	= cv2.resize(frame_left, (480, 270))
		
		# Show the results
		#cv2.imshow("Combined Cameras", np.hstack((frame0_ud, frame1_ud)) )
		cv2.imshow("Left Camera / Average of cameras", np.hstack((frame_left, frame_avg)) )
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap0.release()
	cap1.release()
	cv2.destroyAllWindows()












