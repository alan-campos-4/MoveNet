import numpy as np
import cv2
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline


# 1) 
data = np.load("params/disp_params_rectified.npz", allow_pickle=True)
map1x = data['map1_l']
map1y = data['map2_l']
map2x = data['map1_r']
map2y = data['map2_r']
Q     = data['Q']


# 2) 
block_size	= 5
num_disp	= 16*6
stereo = cv2.StereoSGBM_create(
    minDisparity	= 0,
    numDisparities	= num_disp,
    blockSize		= block_size,
    P1				= 8*1*block_size**2,
    P2				= 32*1*block_size**2,
    uniquenessRatio	= 10,
    speckleWindowSize= 100,
    speckleRange	= 32,
    disp12MaxDiff	= 1
)


# 3) 
capL = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
if not capL.isOpened() or not capR.isOpened():
    print("Camera is not open!")
    sys.exit(1)


# 4) 
while True:
    retL, fL = capL.read()
    retR, fR = capR.read()
    if not retL or not retR:
        break

    # a)
    rL = cv2.remap(fL, map1x, map1y, cv2.INTER_LINEAR)
    rR = cv2.remap(fR, map2x, map2y, cv2.INTER_LINEAR)

    # b)
    gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)

    # c)
    disp = stereo.compute(gL, gR).astype(np.float32) / 16.0
    disp = np.clip(disp, 0, num_disp)

    # d)
    pts3d =		cv2.reprojectImageTo3D(disp, Q)
    depth_map =	pts3d[:, :, 2]

    # e)
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imshow('Disparity', cv2.applyColorMap(disp_vis,cv2.COLORMAP_JET))

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


capL.release()
capR.release()
cv2.destroyAllWindows()
