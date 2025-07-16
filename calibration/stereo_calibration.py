import numpy as np
import cv2
import glob

# Define number of inner corners in your chessboard pattern (width-1, height-1)
nx, ny = 9, 6  # 10x7 squares => 8x5 internal corners

# Prepare object points like (0,0,0), (1,0,0), (2,0,0), ..., (7,4,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []      # 3d points in real world space
imgpointsL = []     # 2d points in left image plane
imgpointsR = []     # 2d points in right image plane

# Load all calibration images from disk
images_left = sorted(glob.glob('captures/left_*.png'))
images_right = sorted(glob.glob('captures/right_*.png'))

imgL = 0
imgR = 0

# Iterate over image pairs to find chessboard corners
for imgL_path, imgR_path in zip(images_left, images_right):
    imgL = cv2.imread(imgL_path, 0)  # Load as grayscale
    imgR = cv2.imread(imgR_path, 0)

    # Try to find chessboard corners in both images
    retL, cornersL = cv2.findChessboardCorners(imgL, (nx, ny), None)
    retR, cornersR = cv2.findChessboardCorners(imgR, (nx, ny), None)

    # If found in both images, save the points
    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

# Calibrate each camera individually
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, imgL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, imgR.shape[::-1], None, None)

# Perform stereo calibration to find relative rotation (R) and translation (T)
retStereo, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR, imgL.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Rectify both cameras (align them for stereo matching)
RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, imgL.shape[::-1], R, T)

# Generate undistortion and rectification maps
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, imgL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, imgR.shape[::-1], cv2.CV_32FC1)

# Save maps and Q matrix for later use
np.savez("stereo_params.npz", mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy, Q=Q)
print("Stereo calibration completed and parameters saved successfully.")
