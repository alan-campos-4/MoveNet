import os
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import numpy as np
import cv2

# Load stereo parameters
data = np.load("params/stereo_params_undistort.npz")


# Extract intrinsics and extrinsics
K1 = data['K1']  # Camera 1 intrinsic matrix
D1 = data['D1']  # Camera 1 distortion coefficients
K2 = data['K2']  # Camera 2 intrinsic matrix
D2 = data['D2']  # Camera 2 distortion coefficients
R =  data['R']   # Rotation matrix between cameras
T =  data['T']   # Translation vector between cameras




# Initialize the cameras (assuming two USB cameras)
cap1 = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)  # Camera 1
cap2 = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)  # Camera 2

# Check if cameras opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Unable to capture frames.")
        break

    # Optional: Undistort the images based on camera parameters
    frame1_undistorted = cv2.undistort(frame1, K1, D1)
    frame2_undistorted = cv2.undistort(frame2, K2, D2)

    # Display the resulting frames
    cv2.imshow('Camera 1', frame1_undistorted)
    cv2.imshow('Camera 2', frame2_undistorted)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap1.release()
cap2.release()
cv2.destroyAllWindows()


