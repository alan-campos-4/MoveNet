def gstreamer_pipeline(sensor_id, width=1280, height=720, framerate=30, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width=1280,height=720,framerate={framerate}/1,format=NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw,format=BGRx,width={width},height={height} ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )
