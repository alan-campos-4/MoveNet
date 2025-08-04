import os
import sys
import time
import numpy as np
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')

from pipeline import gstreamer_pipeline
from pose_estimation import *
from timestamps import *
from tflite_c_runner import TFLiteCModel 

if __name__ == '__main__':
    model = TFLiteCModel(
        model_path='models/movenet-thunder.tflite',
        lib_path='/home/jetson_0/Documents/tflite_c_build/libtensorflowlite_c.so'
    )

    cap = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: couldn't open the camera.")
        exit()

    start_time = time.time()
    frame_count = 0
    fps = 0
    fps_array = []
    seconds_passed = 0
    max_seconds = get_max_seconds()

    try:
        while seconds_passed < max_seconds:
            ret, frame = cap.read()
            if not ret:
                print("Error: can't receive frame.")
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                seconds_passed += 1
                fps = frame_count / elapsed_time
                fps_array.append(fps)
                frame_count = 0
                start_time = time.time()

            img = frame.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
            input_image = tf.cast(img, dtype=tf.float32)

            keypoints_with_scores = model.predict(input_image.numpy())

            draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
            draw_keypoints(frame, keypoints_with_scores, 0.4)
            show_text(cv2, frame, seconds_passed, max_seconds, fps)

            cv2.imshow('Pose Estimation (C API)', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        save_performance(__file__, 'Pose estimation (C API)', fps_array, max_seconds, cap)

    except KeyboardInterrupt as e:
        print(e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
