import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # enables more tf instructions in operations
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
from camera_thread import *
from pose_estimation import *
import cv2
import numpy as np
import tensorflow as tf
import vpi
import time


WINDOW_SIZE = 9
MAX_DISP = 128


"""
	1.	Run MoveNet on the RGB image from both cameras.
	2.	Extract the keypoints.
	3.	Create the disparity map from both cameras.
	4.	Match the coordinates of these keypoints with the corresponding locations in the disparity map.
	5.	Estimate the depth/distance at these keypoints using the disparity values.
"""

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Preprocess function using OpenCV CUDA
def preprocess(img):
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(img)
    resized_gpu = cv2.cuda.resize(gpu_mat, (256, 256))
    img_resized = resized_gpu.download()
    img_resized = img_resized.astype(np.int32)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized


if __name__ == '__main__':

	# Load MoveNet model
	model = tf.saved_model.load("models/singlepose_thunder_tf")
	movenet = model.signatures["serving_default"]

	# Load stereo calibration maps
	map_l, map_r = get_calibration()
	cam_l = CameraThread(0)
	cam_r = CameraThread(1)
	
	# Upload remap matrices to GPU once
	map_l_x_gpu = cv2.cuda_GpuMat()
	map_l_y_gpu = cv2.cuda_GpuMat()
	map_r_x_gpu = cv2.cuda_GpuMat()
	map_r_y_gpu = cv2.cuda_GpuMat()
	map_l_x_gpu.upload(map_l[0])
	map_l_y_gpu.upload(map_l[1])
	map_r_x_gpu.upload(map_r[0])
	map_r_y_gpu.upload(map_r[1])
	
    # Create Gaussian filter for CUDA
	gauss_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 0)

    # Skip first few frames (warmup)
	time.sleep(0.5)
	for _ in range(5):
		_ = cam_l.image
		_ = cam_r.image
		time.sleep(0.05)
        
	try:
		with vpi.Backend.CUDA:
			frame_count = 0
			while True:
				frame_start_time = time.time()  # Start measuring time for FPS
				
				# Get images from cameras
				arr_l = cam_l.image
				arr_r = cam_r.image
				
				# Upload images to GPU
				cam_l_gpu = cv2.cuda_GpuMat()
				cam_r_gpu = cv2.cuda_GpuMat()
				cam_l_gpu.upload(arr_l)
				cam_r_gpu.upload(arr_r)
                
				# Rectify using remap on GPU
				arr_rect_0_gpu = cv2.cuda.remap(cam_l_gpu, map_l_x_gpu, map_l_y_gpu, interpolation=cv2.INTER_LINEAR)
				arr_rect_1_gpu = cv2.cuda.remap(cam_r_gpu, map_r_x_gpu, map_r_y_gpu, interpolation=cv2.INTER_LINEAR)
				
				# Apply Gaussian blur on GPU
				arr_rect_0_gpu = gauss_filter.apply(arr_rect_0_gpu)
				arr_rect_1_gpu = gauss_filter.apply(arr_rect_1_gpu)

               # Download to CPU for inference
				arr_rect_0 = arr_rect_0_gpu.download()
				arr_rect_1 = arr_rect_1_gpu.download()
                
				# Run MoveNet inference on both images
				with tf.device('/GPU:0'):
					print("🧠 Running MoveNet on GPU...")

					input_0 = preprocess(arr_rect_0)
					start = time.time()
					outputs_0 = movenet(input=tf.convert_to_tensor(input_0, dtype=tf.int32))
					inference_time = time.time() - start
					keypoints_0 = outputs_0["output_0"].numpy()
					
					#print("Shape:", input_0.shape)
					#print("Dtype:", input_0.dtype)
					#print(f"Inference time: {inference_time*1000:.2f} ms")
					
					input_1 = preprocess(arr_rect_1)
					outputs_1 = movenet(input=tf.convert_to_tensor(input_1, dtype=tf.int32))
					keypoints_1 = outputs_1["output_0"].numpy()
				
				# Resize images for VPI stereo processing
				arr_rect_0 = cv2.resize(arr_rect_0, (480, 270))
				arr_rect_1 = cv2.resize(arr_rect_1, (480, 270))

				# Convert to VPI images and compute disparity
				vpi_l = vpi.asimage(arr_rect_0).convert(vpi.Format.U16, scale=1)
				vpi_r = vpi.asimage(arr_rect_1).convert(vpi.Format.U16, scale=1)
                
				disparity_16bpp = vpi.stereodisp(
					vpi_l,
					vpi_r,
					out_confmap = None,
					backend = vpi.Backend.CUDA,
					window = WINDOW_SIZE,
					maxdisp = MAX_DISP,
				)
				disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32*MAX_DISP) )
				disp_arr = disparity_8bpp.cpu()
				disp_arr = cv2.medianBlur(disp_arr, 5) # Consider replacing with VPI later
				disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
				
				
				# Overlay keypoints and connections on disparity map
				draw_img = disp_arr.copy()
				draw_connections(draw_img,	keypoints_0, EDGES, 0.4)
				draw_keypoints(draw_img,	keypoints_1, 0.4)
				
				
				#data = np.load("params/disp_params_rectified.npz")
				#K = data["K1"]
				#print("Focal length (fx):", K[0, 0])
		  
				#T = data["T"]
				#baseline_m = abs(T[0])
				#baseline_cm = baseline_m * 100
				#print("Baseline:", baseline_cm, "cm")

				# Estimate depth at keypoints
				focal_length = 752.90670806571
				baseline_cm = 7.74058794
				
				keypoints = keypoints_0[0][0]
				disp_raw = disparity_16bpp.cpu().view(np.ndarray)
				
				for i, (x, y, conf) in enumerate(keypoints):
					if conf > 0.2:
						x_disp = int(x * disp_raw.shape[1])
						y_disp = int(y * disp_raw.shape[0])
						if 0 <= x_disp < disp_raw.shape[1] and 0 <= y_disp < disp_raw.shape[0]:
							disparity_val = disp_raw[y_disp, x_disp]
							
							# Ignore too small disparities (which cause huge depth)
							if disparity_val > 5:
								real_disparity = disparity_val / 32.0
								Z = (focal_length * baseline_cm) / real_disparity

								print(f"Keypoint {i} depth: {Z:.1f} cm")
								cv2.circle(draw_img, (x_disp, y_disp), 4, (0, 255, 255), -1)
								cv2.putText(draw_img, f"{int(Z)} cm", (x_disp + 5, y_disp - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				
				# Show result every 5 frames
				frame_count += 1
				if frame_count % 5 == 0:
					cv2.imshow("Depth-annotated keypoints", draw_img)
				
				frame_end_time = time.time()
				frame_duration = frame_end_time - frame_start_time
				fps = 1.0 / frame_duration if frame_duration > 0 else 0
				print(f"⏱ FPS: {fps:.2f}")


				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	except KeyboardInterrupt as e:
		print(e)
	finally:
		cam_l.stop()
		cam_r.stop()
		cv2.destroyAllWindows()
