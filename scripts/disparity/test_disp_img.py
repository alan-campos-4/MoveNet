import cv2
import numpy as np
from matplotlib import pyplot as plt



if __name__ == '__main__':

	imgL = cv2.imread("./exampleL.png",0)
	imgR = cv2.imread("./exampleR.png",0)
	
	if imgL is None or imgR is None:
		raise FileNotFoundError("One or both images could not be loaded.")
	
	#if imgL.shape != imgR.shape:
		#imgR = cv2.resize(imgR, (imgL.shape[1], imgL.shape[0]))
	
	block_size = 15
	#stereo = cv2.StereoBM(numDisparities=16, blockSize=15)	
	stereo = cv2.StereoSGBM_create(
		minDisparity = 0,
		numDisparities = 16*6,
		blockSize = block_size,
		P1 = 8*1*block_size ** 2,
		P2 = 32*1*block_size ** 2,
		disp12MaxDiff = 1,
		uniquenessRatio = 10,
		speckleWindowSize = 100,
		speckleRange = 32
	)
	disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
	
	plt.imshow(disparity, 'gray')
	plt.title('Disparit Map')
	plt.show()
