import cv2
import numpy as np
import vpi
import time
from datetime import datetime
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline


MAX_DISP = 128
WINDOW_SIZE	= 10

# Load rectification maps (adjust if using .npz or .npy)
data = np.load("params/disp_params_rectified.npz")
map_l = (data["map1_l"], data["map2_l"])
map_r = (data["map1_r"], data["map2_r"])

# Initialize left and right CSI cameras using GStreamer
cap_left = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
cap_right = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

print("Press 's' to save the current disparity map, 'q' to quit.")

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not (ret_l and ret_r):
        print("Failed to capture frames from one or both cameras.")
        break

    # Rectify both frames using the calibration maps
    frame_l_rect = cv2.remap(frame_l, *map_l, cv2.INTER_LANCZOS4)
    frame_r_rect = cv2.remap(frame_r, *map_r, cv2.INTER_LANCZOS4)

    # Resize the rectified images to reduce processing load
    frame_l_rect = cv2.resize(frame_l_rect, (480, 270))
    frame_r_rect = cv2.resize(frame_r_rect, (480, 270))

    # Run VPI backend on CUDA for performance
    with vpi.Backend.CUDA:
        gray_l = cv2.cvtColor(frame_l_rect, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r_rect, cv2.COLOR_BGR2GRAY)
        # Convert OpenCV images to VPI images in 16-bit format
        vpi_l = vpi.asimage(frame_l_rect).convert(vpi.Format.U16, scale=1)
        vpi_r = vpi.asimage(frame_r_rect).convert(vpi.Format.U16, scale=1)

        # Compute disparity map using VPI's stereo disparity function
        disparity_16bpp = vpi.stereodisp(
            vpi_l, vpi_r,
            backend=vpi.Backend.CUDA,
            window=WINDOW_SIZE,
            maxdisp=MAX_DISP
        )

        # Convert the disparity map to 8-bit format for visualization
        disparity_8bpp = disparity_16bpp.convert(
            vpi.Format.U8, scale=255.0 / (32 * MAX_DISP)
        )

        # Move the image from GPU to CPU memory
        disp_arr = disparity_8bpp.cpu()

    # Apply a color map to the disparity image for better visualization
    disp_color = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)

    # Show the colored disparity map
    cv2.imshow("Disparity", disp_color)

    # Keyboard controls: 'q' to quit, 's' to save the current frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"depth_{timestamp}.jpg", disp_color)
        print(f"Saved: depth_{timestamp}.jpg")


# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

