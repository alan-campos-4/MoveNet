def gstreamer_pipeline(sensor_id, width=1280, height=720, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width=3264,height=2464,framerate=30/1,format=NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw,format=BGRx,width={width},height={height} ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true"
    )

# Örnek kullanım:
pipeline = gstreamer_pipeline(sensor_id=0)
print(pipeline)
