import os
import cv2
from pipeline import gstreamer_pipeline 

os.makedirs("captures", exist_ok=True)

capL = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

counter = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)

    key = cv2.waitKey(1)
    if key == ord('c'):  # capture
        cv2.imwrite(f"captures/left_{counter}.png", frameL)
        cv2.imwrite(f"captures/right_{counter}.png", frameR)
        counter += 1
        print("Captured image pair", counter)
    elif key == ord('q'):
        break
