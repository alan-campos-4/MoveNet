import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pipeline import gstreamer_pipeline




#stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=7,
    P1=8*3*7**2,
    P2=32*3*7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)


if __name__ == '__main__':

    cap0 = cv.VideoCapture(gstreamer_pipeline(0), cv.CAP_GSTREAMER)
    cap1 = cv.VideoCapture(gstreamer_pipeline(1), cv.CAP_GSTREAMER)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

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
        disparity[disparity < 0] = 0
        
        disp_norm = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)
        
        disp_clahe = clahe.apply(disp_norm)
        disp_final = cv.GaussianBlur(disp_clahe, (3, 3), 0)

        cv.imshow('Clean Disparity Map', disp_final)
        #cv.imshow('Left Cam',  frame1)
        #cv.imshow('Right Cam', frame0)

        if cv.waitKey(1) & 0xFF==ord('q'):
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()


