import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pose_estimation import *
from pipeline import gstreamer_pipeline
import numpy as np
import cv2
import numpy as np
import time


"""		timestamps test_2.py
Basic test to meassure the camera feed framerate of one camera. 
Performs camera feed calibration.
"""



if __name__ == '__main__':
	
	# Opens the camera
	cap = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
	if (cap.isOpened()==False):
		print("Error: couldn't open the camera.")
		exit()
	
	# Meassurement variables
	frame_count = 0
	start_time = time.time()
	fps_rec = []
	font = cv2.FONT_HERSHEY_PLAIN
	
	# Calibration parameters
	data = np.load("params/stereo_params_undistort.npz")
	K1 = data['K1']
	D1 = data['D1']
	
	while True:		
		ret, frame = cap.read()
		if not ret:
			print("Error: can't receive frame.")
			break
		
		# Adds the time passed
		frame_count += 1
		elapsed_time = time.time()-start_time
		
		# Calculates the frames per second in the time passed
		if elapsed_time > 1.0:
			fps = frame_count / elapsed_time
			fps_rec.append(fps)
			print(f"FPS: {fps:.2f}")
			frame_count = 0
			start_time = time.time()
		
		# Shows the feed with framerate
		cv2.putText(frame, str(fps)+' FPS', (20,40), font, 2, (255,255,255), 2, cv2.LINE_AA)
		cv2.imshow('Timestamps', frame)
		
		# Break the loop if the 'Q' key is pressed
		if cv2.waitKey(10) & 0xFF==ord('q'):
			break
		
	cap.release()
	cv2.destroyAllWindows()
	
	# Saves the results
	with open("output/timestamps_test_2.txt", "w") as f:
		f.write("Camera feed calibration.\n")
		f.write('--- ',str(datetime.now()), ' ---\n')
		for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
			f.write('Frame %d difference:'%i, abs(ts - cts))
			f.write('\n')
	print('Performance saved to file.')"""	


