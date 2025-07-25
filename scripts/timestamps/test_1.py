import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import numpy as np
import cv2
import time


"""		timestamps test_1.py
Basic test to meassure the camera feed framerate of one camera. 
Regular feed with no extra calculations or calibration.
"""



if __name__ == '__main__':
	
	# Opens the camera
	cap = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
	if (cap.isOpened()==False):
		print("Error: couldn't open the camera.")
		exit()
	
	# Meassurement variables
	start_time = time.time()
	frame_count = 0
	fps = 0
	fps_rec = []
	font = cv2.FONT_HERSHEY_PLAIN
	
	try:
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
				frame_count = 0
				start_time = time.time()
			
			# Shows the feed with framerate
			cv2.putText(frame, str(fps)+' FPS', (20,40), font, 2, (255,255,255), 2, cv2.LINE_AA)
			cv2.imshow('Timestamps', frame)
			
			# Break the loop if the 'Q' key is pressed
			if cv2.waitKey(10) & 0xFF==ord('q'):
				break
			
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cap.release()
		cv2.destroyAllWindows()
	
	# Saves the results
	with open("output/timestamps/test_1.txt", "w") as f:
		f.write("Regular feed.\n")
		for fps in fps_rec:
			f.write(f'FPS: {fps}')
			f.write('\n')
	print('Performance saved to file.')

