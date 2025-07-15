import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # enables more tf instructions in operations
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pipeline import gstreamer_pipeline



# Load calibration parameters
data = np.load("stereo_params.npz")
mapLx = data["mapLx"]
mapLy = data["mapLy"]
mapRx = data["mapRx"]
mapRy = data["mapRy"]

data = np.load("stereo_params_1.npz")
for key in data.files:
    print(f"{key}: {data[key].shape}")






calib_Lheight, calib_Lwidth = mapLx.shape
calib_Rheight, calib_Rwidth = mapRx.shape


if __name__ == '__main__':

    # Open both cameras
    cap0 = cv2.VideoCapture(gstreamer_pipeline(0, calib_Lwidth, calib_Lheight), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(gstreamer_pipeline(1, calib_Rwidth, calib_Rheight), cv2.CAP_GSTREAMER)

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
        rectifiedL = cv2.remap(frame0, mapLx, mapLy, cv2.INTER_LINEAR)
        rectifiedR = cv2.remap(frame1, mapRx, mapRy, cv2.INTER_LINEAR)
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

