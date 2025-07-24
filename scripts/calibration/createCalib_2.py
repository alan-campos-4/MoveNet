import cv2
import numpy as np
import glob




# Chessboard settings
chessboard_size = (9, 6)
square_size = 0.025  # 2.5cm per square (adjust as needed)

# Termination criteria
criteria		= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo	= (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# Prepare object points
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays for points
objpoints =  []
imgpointsL = []
imgpointsR = []

imagesL = sorted(glob.glob("img/captures/left_*.png"))
imagesR = sorted(glob.glob("img/captures/right_*.png"))

for imgL_path, imgR_path in zip(imagesL, imagesR):
    imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print("Error: Could not open or find the images!")
        continue

    print(f"New images. L:{imgL_path},{imgL.shape} | R:{imgR_path},{imgR.shape}")

    retL, cornersL = cv2.findChessboardCorners(imgL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(imgR, chessboard_size, None)

    print(f"Corners found: Left: {retL}, Right: {retR}")  # Add this line to log results

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)

        # Draw and save the chessboard corners
        cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)
        
        cv2.imwrite(f"output/cornered_{imgL_path.split('/')[-1]}", imgL)
        cv2.imwrite(f"output/cornered_{imgR_path.split('/')[-1]}", imgR)

        cv2.imshow('Left', imgL)
        cv2.imshow('Right', imgR)
        cv2.waitKey(100)  # Display each pair for a short time

    print("Object points length: ", len(objpoints))
    print("Number of left images:  ", len(imgpointsL))
    print("Number of right images: ", len(imgpointsR))


cv2.destroyAllWindows()

if len(imgpointsL)!=len(imgpointsR):
	print("Error: unequal number of images.")
	exit()

# Calibrate each camera individually
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, imgL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, imgR.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    imgL.shape[::-1], criteria=criteria_stereo, flags=flags
)

# Generate undistortion and rectification maps
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, imgL.shape[::-1], R, T)
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, imgL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, imgR.shape[::-1], cv2.CV_32FC1)

# Save the parameters
np.savez("params/stereo_params_2.npz",
         mapLx=mapLx, mapLy=mapLy,
         mapRx=mapRx, mapRy=mapRy,
         Q=Q)

print("Stereo calibration complete and saved.")
