import cv2
import threading

def gstreamer_pipeline(sensor_id):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=true"
    )

def show_camera(sensor_id, window_name):
    cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Error: couldn't open camera {sensor_id}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: couldn't read frame from camera {sensor_id}")
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


# Run both camera streams in parallel using threads
if __name__=='__main__':

    cam0_thread = threading.Thread(target=show_camera, args=(0, "Camera 0"))
    cam1_thread = threading.Thread(target=show_camera, args=(1, "Camera 1"))

    cam0_thread.start()
    cam1_thread.start()

    cam0_thread.join()
    cam1_thread.join()

