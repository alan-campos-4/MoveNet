import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import numpy as np
import cv2
import time


"""
Meassures the framerate of two camera feeds simultaneously after calibration.
"""



if __name__ == '__main__':
	
	# Opens the camera
	cap0 = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
	cap1 = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
	if (cap0.isOpened()==False) or (cap1.isOpened()==False):
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
	K2 = data['K2']
	D2 = data['D2']
	
	try:
		while seconds_passed < max_seconds:		
			ret0, frame0 = cap0.read()
			ret1, frame1 = cap1.read()
			if not ret0 or not ret1:
				print("Error: can't receive frame.")
				break
			
			# Calibrates the camera
			frame0 = cv2.undistort(frame0, K1, D1)
			frame1 = cv2.undistort(frame1, K2, D2)
			
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
			
			# Shows the feed with framerate
			show_text(cv2, frame0, seconds_passed, max_seconds, fps)
			cv2.imshow('Timestamps', np.hstack((frame0, frame1)) )
			
			# Break the loop if the 'Q' key is pressed
			if cv2.waitKey(10) & 0xFF==ord('q'):
				break
			
		# Saves the results
		save_performance(__file__, 'Display and calibrate both cameras', fps_array, max_seconds, cap0, cap1)
		
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cap0.release()
		cap1.release()
		cv2.destroyAllWindows()
	
