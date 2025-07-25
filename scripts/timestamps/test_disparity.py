import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from camera_thread import *
import numpy as np
import cv2
import time


"""		timestamps test 3.py
Meassures the framerate of two cameras while performing 
camera feed calibration and disparity calculation
"""




if __name__ == '__main__':
	
	# Opens the camera
	map_l, map_r = get_calibration()
	cam_l = CameraThread(0)
	cam_r = CameraThread(1)
	
	# Meassurement variables
	start_time = time.time()
	frame_count = 0
	fps = 0
	fps_rec = []
	font = cv2.FONT_HERSHEY_PLAIN
	
	try:
		with vpi.Backend.CUDA:
			while True:
				arr_l = cam_l.image
				arr_r = cam_r.image
				#arr_l = cv2.undistort(arr_l, K1, D1)
				#arr_r = cv2.undistort(arr_r, K2, D2)
				
				# Rectify
				arr_l_rect = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
				arr_r_rect = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)
				
				# Apply slight Gaussian blur to reduce high-frequency noise
				arr_l_rect = cv2.GaussianBlur(arr_l_rect, (3, 3), 0)
				arr_r_rect = cv2.GaussianBlur(arr_r_rect, (3, 3), 0)
				
				# Resize
				arr_l_rect = cv2.resize(arr_l_rect, (480, 270))
				arr_r_rect = cv2.resize(arr_r_rect, (480, 270))
				
				# Convert to VPI image
				vpi_l = vpi.asimage(arr_l_rect)
				vpi_r = vpi.asimage(arr_r_rect)
				
				vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
				vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
				
				disparity_16bpp = vpi.stereodisp(
					vpi_l_16bpp,
					vpi_r_16bpp,
					out_confmap = None,
					backend = vpi.Backend.CUDA,
					window = WINDOW_SIZE,
					maxdisp = MAX_DISP,
				)
				disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32*MAX_DISP) )
				disp_arr = disparity_8bpp.cpu()
				disp_arr = cv2.medianBlur(disp_arr, 5)
				disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
				
				# Adds the time passed
				frame_count += 1
				elapsed_time = time.time()-start_time
				# Calculates the frames per second in the time passed
				if elapsed_time > 1.0:
					fps = frame_count / elapsed_time
					fps_rec.append(fps)
					frame_count = 0
					start_time = time.time()
				
				# Break the loop if the 'Q' key is pressed
				if cv2.waitKey(10) & 0xFF==ord('q'):
					break
				
				# Shows the feed with framerate
				cv2.putText(disp_arr, str(fps)+' FPS', (20,40), font, 2, (255,255,255), 2, cv2.LINE_AA)
				cv2.imshow('Timestamps', disp_arr)
				
	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
		cv2.destroyAllWindows()
	
	# Saves the results
	with open("output/timestamps/test_disparity.txt", "w") as f:
		f.write("Camera calibration and disparity.\n")
		for fps in fps_rec:
			f.write(f'FPS: {fps}')
			f.write('\n')
	print('Performance saved to file.')

