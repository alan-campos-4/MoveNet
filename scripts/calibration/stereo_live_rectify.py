import cv2
import numpy as np

# Load stereo calibration parameters
data = np.load("stereo_params.npz")
mapLx = data["mapLx"]
mapLy = data["mapLy"]
mapRx = data["mapRx"]
mapRy = data["mapRy"]

# Open left and right CSI cameras via GStreamer
gst_left = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
gst_right = 'nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

capL = cv2.VideoCapture(gst_left, cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gst_right, cv2.CAP_GSTREAMER)

if not (capL.isOpened() and capR.isOpened()):
    print("Failed to open one or both cameras.")
    exit()

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Failed to read frame")
        break

    # Rectify both frames
    rectifiedL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Optional: Resize for display
    displayL = cv2.resize(rectifiedL, (640, 360))
    displayR = cv2.resize(rectifiedR, (640, 360))

    combined = cv2.hconcat([displayL, displayR])
    cv2.imshow("Rectified Stereo Cameras", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
