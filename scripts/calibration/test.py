import os
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import cv2
import numpy as np
from matplotlib import pyplot as plt



# Load calibration parameters
data = np.load("params/stereo_params_4.npz")
map1_l = data["map1_l"]
map2_l = data["map2_l"]
map1_r = data["map1_r"]
map2_r = data["map2_r"]




if __name__ == '__main__':

    # Open both cameras
    cap0 = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

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

        # Rectify both frames and resize
        rectifiedL = cv2.remap(frame0, map1_l, map2_l, cv2.INTER_LINEAR)
        rectifiedR = cv2.remap(frame1, map1_r, map2_r, cv2.INTER_LINEAR)
        displayL = cv2.resize(rectifiedL, (640, 360))
        displayR = cv2.resize(rectifiedR, (640, 360))

        # Combine the frames horizontally
        combined = np.hstack((rectifiedL, rectifiedR))

        # Show in one window
        cv2.imshow("Combined MoveNet Thunder", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

