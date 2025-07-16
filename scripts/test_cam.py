import cv2

gst_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink max-buffers=1 drop=true"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



while True:
    if not cap.isOpened():
        print("Error: couldn't open the camera.")
        break

    ret, frame = cap.read()
    if not ret:
        print("Error: couldn't read frame.")
        break

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

