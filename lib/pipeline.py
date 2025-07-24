
def gstreamer_pipeline(sensor_id, width=1280, height=720):
    return (
        f"nvarguscamerasrc sensor_id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate=30/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=true"
    )


#  "nvarguscamerasrc sensor-id=1 ! "
#  "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
#  "nvvidconv flip-method=0 ! "
#  "video/x-raw, width=1280, height=720, format=BGRx ! "
#  "videoconvert ! "
#  "video/x-raw, format=BGR ! appsink max-buffers=1 drop=true"


