# main_depth_with_gpu.py
import os
import sys
import time
import cv2
import numpy as np
import vpi
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')

from pipeline import gstreamer_pipeline
from camera_thread import *
from pose_estimation import *
from tflite_c_runner import TFLiteCModel

if __name__ == '__main__':
    # Load the C API model
    model = TFLiteCModel(
        model_path='models/movenet-thunder.tflite',
        lib_path='/home/jetson_0/Documents/tflite_c_build/libtensorflowlite_c.so'
    )

    # Open both cameras
    map_l, map_r = get_calibration()
    cam_l = CameraThread(0)
    cam_r = CameraThread(1)
    time.sleep(0.5)
    for _ in range(5):
        _ = cam_l.image
        _ = cam_r.image
        time.sleep(0.05)

    try:
        with vpi.Backend.CUDA:
            while True:
                arr_l = cam_l.image
                arr_r = cam_r.image
                for _ in range(5):
                    _ = cam_l.image
                    _ = cam_r.image

                arr_rect_0 = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
                arr_rect_1 = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)

                img0 = tf.image.resize_with_pad(np.expand_dims(arr_rect_0, axis=0), 256, 256)
                img1 = tf.image.resize_with_pad(np.expand_dims(arr_rect_1, axis=0), 256, 256)
                input_image_0 = tf.cast(img0, dtype=tf.float32)
                input_image_1 = tf.cast(img1, dtype=tf.float32)

                keypoints_0 = model.predict(input_image_0.numpy())
                keypoints_1 = model.predict(input_image_1.numpy())

                arr_rect_0 = cv2.GaussianBlur(arr_rect_0, (3, 3), 0)
                arr_rect_1 = cv2.GaussianBlur(arr_rect_1, (3, 3), 0)
                arr_rect_0 = cv2.resize(arr_rect_0, (480, 270))
                arr_rect_1 = cv2.resize(arr_rect_1, (480, 270))

                vpi_l = vpi.asimage(arr_rect_0)
                vpi_r = vpi.asimage(arr_rect_1)
                vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
                vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
                disparity_16bpp = vpi.stereodisp(
                    vpi_l_16bpp, vpi_r_16bpp, out_confmap=None,
                    backend=vpi.Backend.CUDA, window=WINDOW_SIZE, maxdisp=MAX_DISP
                )
                disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0 / (32 * MAX_DISP))
                disp_arr = disparity_8bpp.cpu()
                disp_arr = cv2.medianBlur(disp_arr, 5)
                disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)

                draw_img = disp_arr.copy()
                draw_connections(draw_img, keypoints_0, EDGES, 0.4)
                draw_keypoints(draw_img, keypoints_1, 0.4)

                focal_length = 752.90670806571
                baseline_cm = 7.74058794
                keypoints = keypoints_0[0][0]
                disp_raw = disparity_16bpp.cpu().view(np.ndarray)

                for i, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.2:
                        x_disp = int(x * disp_raw.shape[1])
                        y_disp = int(y * disp_raw.shape[0])
                        if 0 <= x_disp < disp_raw.shape[1] and 0 <= y_disp < disp_raw.shape[0]:
                            disparity_val = disp_raw[y_disp, x_disp]
                            if disparity_val > 0:
                                real_disparity = disparity_val / 32.0
                                Z = (focal_length * baseline_cm) / real_disparity
                                print(f"Keypoint {i} depth: {Z:.1f} cm")
                                cv2.circle(draw_img, (x_disp, y_disp), 4, (0, 255, 255), -1)
                                cv2.putText(draw_img, f"{int(Z)} cm", (x_disp + 5, y_disp - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.imshow("Depth-annotated keypoints", draw_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cam_l.stop()
        cam_r.stop()
        cv2.destroyAllWindows()
