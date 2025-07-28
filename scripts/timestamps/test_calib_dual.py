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
	
	# Establish time limit for test
	MAX_SECONDS = 20
	if (len(sys.argv)==2):
		if (type(sys.argv[1])==type(20)):
			MAX_SECONDS = int(sys.argv[1])
	print(f'Test will last for {MAX_SECONDS} seconds')
	
	# Meassurement variables
	start_time = time.time()
	seconds_passed = 0
	frame_count = 0
	fps = 0
	fps_rec = []
	font = cv2.FONT_HERSHEY_PLAIN
	
	# Calibration parameters
	data = np.load("params/stereo_params_undistort.npz")
	K1 = data['K1']
	D1 = data['D1']
	K2 = data['K2']
	D2 = data['D2']
	
	try:
		while seconds_passed < MAX_SECONDS:		
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
				fps_rec.append(fps)
				frame_count = 0
				start_time = time.time()
			
			# Shows the feed with framerate
			cv2.putText(frame0, f'Second = {seconds_passed} | FPS = {fps}', (20,40), font, 2, (255,255,255), 2, cv2.LINE_AA)
			cv2.imshow('Timestamps', np.hstack((frame0, frame1)) )
			
			# Break the loop if the 'Q' key is pressed
			if cv2.waitKey(10) & 0xFF==ord('q'):
				break
		
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cap.release()
		cv2.destroyAllWindows()
	
	# Saves the results
	with open("output/timestamps/test_calib_dual.txt", "w") as f:
		f.write(f"Display and calibration of 2 cameras during {MAX_SECONDS}.\n")
		for fps in fps_rec:
			f.write(f'FPS: {fps}')
			f.write('\n')
	print('Performance saved to file.')

