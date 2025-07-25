import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline



if __name__ == '__main__':

	cap = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

	data = np.load("params/stereo_params_undistort.npz")
	K1 = data['K1']
	D1 = data['D1']

	while True:
		if (cap.isOpened()==False):
			print("Error: couldn't open the camera.")
			break
		ret0, frame0 = cap.read()
		ret1, frame1 = cap.read()
		if not ret0 or not ret1:
			print("Error: can't receive frame.")
			break

		# Correct distortion
		frame1 = cv2.undistort(frame1, K1, D1)

		# Combine the frames horizontally
		frame0 = cv2.resize(frame0, (480, 270))
		frame1 = cv2.resize(frame1, (480, 270))
		combined = np.hstack((frame0, frame1))

		# Show in one window
		cv2.imshow("Normal / Corrected Feed", combined)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

