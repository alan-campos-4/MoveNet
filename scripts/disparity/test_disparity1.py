import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline


block_size=15
stereo = cv.StereoSGBM_create(
	minDisparity=0,
	numDisparities=16*6,
	blockSize=block_size,
	P1=8 * 1 * block_size ** 2,
	P2=32 * 1 * block_size ** 2,
	disp12MaxDiff=1,
	uniquenessRatio=10,
	speckleWindowSize=100,
	speckleRange=32
)



if __name__ == '__main__':

    cap0 = cv.VideoCapture(gstreamer_pipeline(0), cv.CAP_GSTREAMER)
    cap1 = cv.VideoCapture(gstreamer_pipeline(1), cv.CAP_GSTREAMER)

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
            print("Error: couldn't read frame.")
            break

        grayL = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        disp_norm = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)

        cv.imshow('Disparity Map', disparity)

        if cv.waitKey(1) & 0xFF==ord('q'):
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()
