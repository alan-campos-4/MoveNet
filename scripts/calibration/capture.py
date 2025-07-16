import os
import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline 


os.makedirs("img/captures", exist_ok=True)

capL = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

counter = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    frameL = cv2.resize(frameL, (640, 360))
    frameR = cv2.resize(frameR, (640, 360))

    combined = np.hstack((frameL, frameR))
    
    cv2.imshow("Capture", combined)

    key = cv2.waitKey(1)
    if key == ord('c'):  # capture
        cv2.imwrite(f"img/captures/left_{counter}.png", frameL)
        cv2.imwrite(f"img/captures/right_{counter}.png", frameR)
        counter += 1
        print("Captured image pair", counter)
    elif key == ord('q'):
        break
