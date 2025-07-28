import os
import cv2
import numpy as np
import glob



# Chessboard settings
chessboard_size = (9, 6)  # inner corners (10x7 squares → 9x6)
square_size = 0.025       # square size in meters

# Termination criteria
criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# Flags for corner detection
cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

# Prepare object points once
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store points
objpoints = []
imgpointsL = []
imgpointsR = []

# Create debug output folder
os.makedirs("img/output", exist_ok=True)

# Load image file names
imagesL = sorted(glob.glob("output/captures/left_*.png"))
imagesR = sorted(glob.glob("output/captures/right_*.png"))

for idx, (lpath, rpath) in enumerate(zip(imagesL, imagesR)):
    imgL = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        print(f"Error: Cannot open {lpath} or {rpath}")
        continue

    print(f"Processing pair #{idx}: {lpath} {imgL.shape}, {rpath} {imgR.shape}")
    retL, cornersL = cv2.findChessboardCorners(imgL, chessboard_size, cb_flags)
    retR, cornersR = cv2.findChessboardCorners(imgR, chessboard_size, cb_flags)
    print(f"Corners found: Left={retL}, Right={retR}")

    if retL and retR:
        cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria_subpix)
        cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria_subpix)

        objpoints.append(objp.copy())
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        # Draw and save debug images
        dbgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        dbgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(dbgL, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(dbgR, chessboard_size, cornersR, retR)
        cv2.imwrite(f"img/output/cornered_left_{idx:02d}.png", dbgL)
        cv2.imwrite(f"img/output/cornered_right_{idx:02d}.png", dbgR)

# Ensure we have enough data
if len(objpoints) == 0 or len(imgpointsL) != len(imgpointsR):
    print("Error: Insufficient valid image pairs for calibration.")
    exit()

# Image size from last valid image
h, w = imgL.shape

# Mono calibration
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, (w, h), None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, (w, h), None, None)
print(f"Mono RMS: Left={retL:.4f}, Right={retR:.4f}")

# Stereo calibration
retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    (w, h),
    criteria=criteria_stereo,
    flags=cv2.CALIB_FIX_INTRINSIC
)
print(f"Stereo RMS: {retS:.4f}")

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, (w, h), R, T, alpha=0
)
map1_l, map2_l = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, (w, h), cv2.CV_16SC2
)
map1_r, map2_r = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, (w, h), cv2.CV_16SC2
)

# Save parameters
os.makedirs("calibration", exist_ok=True)
np.savez(
    "params/stereo_params_3.npz",
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R1=R1, R2=R2, P1=P1, P2=P2,
    map1_l=map1_l, map2_l=map2_l,
    map1_r=map1_r, map2_r=map2_r,
    Q=Q
)
print("Calibration data saved to calibration/stereo_params.npz")
