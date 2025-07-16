import cv2
import numpy as np
from pipeline import gstreamer_pipeline


# Open both cameras
cap0 = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

if not cap0.isOpened():
    print("Error: Could not open camera 0")
    exit()
if not cap1.isOpened():
    print("Error: Could not open camera 1")
    exit()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Error: Could not read from one or both cameras.")
        break

    # Optional: resize for display if needed
    frame0 = cv2.resize(frame0, (960, 540))
    frame1 = cv2.resize(frame1, (960, 540))

    # Combine the frames horizontally
    combined = np.hstack((frame0, frame1))

    # Show in one window
    cv2.imshow("Combined Camera View", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()

