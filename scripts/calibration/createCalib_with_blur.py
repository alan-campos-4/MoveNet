import os
import cv2
import glob
import numpy as np

# Stereo calibration script with robust corner detection and preprocessing
# Requires captures/left_*.png and captures/right_*.png
# Minimum valid pairs: 10

# Chessboard settings: interior corners for a 10×7 squares board
chessboard_size = (9, 6)  # width, height
square_size = 0.0385      # square size in meters

# Termination criteria
criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# Flags to improve corner detection
find_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

# Prepare object points grid
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Containers for calibration data
objpoints = []  # 3D points
imgpointsL = [] # 2D points left
imgpointsR = [] # 2D points right

# Create debug output folder
os.makedirs("img/output", exist_ok=True)

# Minimum required valid pairs
MIN_PAIRS = 10

# Ensure captures folder exists and consistent image counts
assert os.path.isdir("img/captures"), "Error: captures folder not found."
imagesL = sorted(glob.glob("img/captures/left_*.png"))
imagesR = sorted(glob.glob("img/captures/right_*.png"))
assert len(imagesL) == len(imagesR), (
    f"Error: unmatched counts L={len(imagesL)}, R={len(imagesR)}."
)

# CLAHE for local contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Process each image pair
for idx, (fL, fR) in enumerate(zip(imagesL, imagesR)):
    imgL = cv2.imread(fL)
    imgR = cv2.imread(fR)
    if imgL is None or imgR is None:
        print(f"Skipping missing pair: {fL}, {fR}")
        continue

    # Convert to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # Histogram equalization and CLAHE
    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)
    # Optional blur to reduce noise
    grayL = cv2.GaussianBlur(grayL, (5,5), 0)
    grayR = cv2.GaussianBlur(grayR, (5,5), 0)

    # Adaptive threshold (optional) to try as binary mask
    thL = cv2.adaptiveThreshold(grayL, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thR = cv2.adaptiveThreshold(grayR, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Try corner detection on grayscale
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, find_flags)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, find_flags)
    # If failed, try binary image
    if not (retL and retR):
        retL, cornersL = cv2.findChessboardCorners(thL, chessboard_size, find_flags)
        retR, cornersR = cv2.findChessboardCorners(thR, chessboard_size, find_flags)

    # Fallback to SB method if still fails
    if not (retL and retR) and hasattr(cv2, 'findChessboardCornersSB'):
        retL, cornersL = cv2.findChessboardCornersSB(grayL, chessboard_size, cv2.CALIB_CB_SYMMETRIC_GRID)
        retR, cornersR = cv2.findChessboardCornersSB(grayR, chessboard_size, cv2.CALIB_CB_SYMMETRIC_GRID)

    print(f"Pair #{idx}: corners found L={retL}, R={retR}")
    if not (retL and retR):
        continue

    # Sub-pixel corner refinement
    cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria_subpix)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria_subpix)

    # Append points
    objpoints.append(objp.copy())
    imgpointsL.append(cornersL)
    imgpointsR.append(cornersR)

    # Draw and save debug visualization
    dbgL = imgL.copy()
    dbgR = imgR.copy()
    cv2.drawChessboardCorners(dbgL, chessboard_size, cornersL, True)
    cv2.drawChessboardCorners(dbgR, chessboard_size, cornersR, True)
    cv2.imwrite(f"img/output/debug_left_{idx:02d}.png", dbgL)
    cv2.imwrite(f"img/output/debug_right_{idx:02d}.png", dbgR)

# Validate minimum pairs
if len(objpoints) < MIN_PAIRS:
    raise RuntimeError(f"Error: only {len(objpoints)} valid pairs found; need at least {MIN_PAIRS}.")

# Determine image size from last processed image
h, w = grayL.shape

# Mono camera calibration
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, (w, h), None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, (w, h), None, None)
print(f"Mono RMS: Left={retL:.3f}, Right={retR:.3f}")

# Stereo calibration
retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    (w, h), criteria=criteria_stereo, flags=cv2.CALIB_FIX_INTRINSIC
)
print(f"Stereo RMS: {retS:.3f}")

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, (w, h), R, T, alpha=0)
map1_l, map2_l = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)

# Save parameters
os.makedirs("params", exist_ok=True)
np.savez("params/stereo_params_blur.npz",
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R1=R1, R2=R2, P1=P1, P2=P2,
    map1_l=map1_l, map2_l=map2_l,
    map1_r=map1_r, map2_r=map2_r,
    Q=Q
)
print("Calibration data saved to params/stereo_params_blur.npz")
