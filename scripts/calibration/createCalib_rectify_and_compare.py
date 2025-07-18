import cv2
import numpy as np
from pipeline import gstreamer_pipeline

# --- Rectify vs Undistort Comparison Script ---
# Displays undistorted (undistort) images on top and stereo-rectified (rectify) images on bottom.

# Load calibration parameters
data = np.load("calibration/stereo_params_blur.npz")
mtx_l, dist_l = data["mtxL"], data["distL"]
mtx_r, dist_r = data["mtxR"], data["distR"]
map1_l, map2_l = data["map1_l"], data["map2_l"]
map1_r, map2_r = data["map1_r"], data["map2_r"]

# Create undistort-only maps (R = I, P = intrinsic matrix)
h, w = map1_l.shape[:2]
mapu_l1, mapu_l2 = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, np.eye(3), mtx_l, (w, h), cv2.CV_16SC2
)
mapu_r1, mapu_r2 = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, np.eye(3), mtx_r, (w, h), cv2.CV_16SC2
)

# Open CSI cameras via GStreamer
capL = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
if not (capL.isOpened() and capR.isOpened()):
    raise RuntimeError("CSI cameras could not be opened")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not (retL and retR):
        break

    # Undistort (original perspective)
    undistL = cv2.remap(frameL, mapu_l1, mapu_l2, cv2.INTER_LINEAR)
    undistR = cv2.remap(frameR, mapu_r1, mapu_r2, cv2.INTER_LINEAR)

    # Stereo rectify (epipolar alignment)
    rectL = cv2.remap(frameL, map1_l, map2_l, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, map1_r, map2_r, cv2.INTER_LINEAR)

    # Display top row: undistort, bottom row: rectify
    top =	cv2.hconcat([undistL, undistR])
    bottom =	cv2.hconcat([rectL, rectR])
    combined =	cv2.vconcat([top, bottom])

    cv2.imshow("Undistort (top) vs Rectify (bottom)", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()
