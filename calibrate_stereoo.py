import os
import cv2
import numpy as np
import glob

# Stereo calibration script with robust corner detection

# Chessboard settings: interior corners for a 10×7 squares board
chessboard_size = (9, 6)  # width, height
square_size = 0.025        # square size in meters

# Termination criteria
criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# Flags to improve corner detection
find_flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FAST_CHECK
)

# Prepare object points grid
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Containers for calibration
objpoints = []  # 3D points
imgpointsL = []  # 2D points left
imgpointsR = []  # 2D points right

# Create debug output folder
os.makedirs("output", exist_ok=True)

# Load image file lists
targetL = sorted(glob.glob("captures/left_*.png"))
targetR = sorted(glob.glob("captures/right_*.png"))

for idx, (fL, fR) in enumerate(zip(targetL, targetR)):
    imgL = cv2.imread(fL)
    imgR = cv2.imread(fR)
    if imgL is None or imgR is None:
        print(f"Skipping missing pair: {fL}, {fR}")
        continue

    # Convert and equalize
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    # Find chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, find_flags)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, find_flags)

    # Fallback to SB method if regular fails
    if not (retL and retR) and hasattr(cv2, 'findChessboardCornersSB'):
        retL, cornersL = cv2.findChessboardCornersSB(grayL, chessboard_size)
        retR, cornersR = cv2.findChessboardCornersSB(grayR, chessboard_size)

    print(f"Pair #{idx}: corners found L={retL}, R={retR}")
    if not (retL and retR):
        continue

    # Refine corner positions\ n    cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria_subpix)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria_subpix)

    # Append for calibration
    objpoints.append(objp.copy())
    imgpointsL.append(cornersL)
    imgpointsR.append(cornersR)

    # Draw and save debug images
    dbgL = imgL.copy()
    dbgR = imgR.copy()
    cv2.drawChessboardCorners(dbgL, chessboard_size, cornersL, retL)
    cv2.drawChessboardCorners(dbgR, chessboard_size, cornersR, retR)
    cv2.imwrite(f"output/debug_left_{idx:02d}.png", dbgL)
    cv2.imwrite(f"output/debug_right_{idx:02d}.png", dbgR)

# Ensure enough pairs
if len(objpoints) < 10:
    print(f"Error: only {len(objpoints)} valid pairs found, need at least 10.")
    exit(1)

# Determine image size
h, w = grayL.shape

# Mono camera calibration
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, (w, h), None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, (w, h), None, None)
print(f"Mono RMS left={retL:.3f}, right={retR:.3f}")

# Stereo calibration
retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    (w, h), criteria=criteria_stereo, flags=cv2.CALIB_FIX_INTRINSIC
)
print(f"Stereo RMS={retS:.3f}")

# Rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, (w, h), R, T, alpha=0
)
map1_l, map2_l = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_16SC2)

# Save parameters
os.makedirs("calib", exist_ok=True)
np.savez(
    "calib/stereo_params.npz",
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R1=R1, R2=R2, P1=P1, P2=P2,
    map1_l=map1_l, map2_l=map2_l,
    map1_r=map1_r, map2_r=map2_r,
    Q=Q
)
print("Calibration data successfully saved.")
