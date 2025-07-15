import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'# turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # enables more tf instructions in operations
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pipeline import gstreamer_pipeline




# Load calibration parameters
data = np.load("stereo_params.npz")
mapLx = data["mapLx"]
mapLy = data["mapLy"]
mapRx = data["mapRx"]
mapRy = data["mapRy"]

# All the connections between keypoints
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c',
}

# Variables
edge_color =  (0,0,255) #Red
point_color = (0,255,0) #Green

# Draw the keypoints as circles in the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped_array:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, point_color, -1)
            tf.config.list_physical_devices('GPU')


# Draw the edges between the coordinates
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1])) # multiplies the keypoints by the dimesions of the frame

    for edge, color in edges.items(): # for every edge, gets the coordinates of the two points and connects them
        p1, p2 = edge
        y1, x1, c1 = shaped_array[p1]
        y2, x2, c2 = shaped_array[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), edge_color, 2)








if __name__ == '__main__':

    # Loads the model from the file.
    interpreter = tf.lite.Interpreter(model_path='movenet-thunder.tflite')
    interpreter.allocate_tensors()

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
        
        # Rectify both frames and resize
        rectifiedL = cv2.remap(frame0, mapLx, mapLy, cv2.INTER_LINEAR)
        rectifiedR = cv2.remap(frame1, mapRx, mapRy, cv2.INTER_LINEAR)
        displayL = cv2.resize(rectifiedL, (640, 360))
        displayR = cv2.resize(rectifiedR, (640, 360))

        # Reshape image
        img0 = displayL.copy()
        img1 = displayR.copy()
        img0 = tf.image.resize_with_pad(np.expand_dims(img0, axis=0), 256, 256)
        img1 = tf.image.resize_with_pad(np.expand_dims(img1, axis=0), 256, 256)
        input_image_0 = tf.cast(img0, dtype=tf.float32)
        input_image_1 = tf.cast(img1, dtype=tf.float32)

        # Setup input and output
        input_details_0 = interpreter.get_input_details()
        input_details_1 = interpreter.get_input_details()
        output_details_0 = interpreter.get_output_details()
        output_details_1 = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details_0[0]['index'], np.array(input_image_0))
        interpreter.set_tensor(input_details_1[0]['index'], np.array(input_image_1))
        interpreter.invoke()
        keypoints_with_scores_0 = interpreter.get_tensor(output_details_0[0]['index'])
        keypoints_with_scores_1 = interpreter.get_tensor(output_details_1[0]['index'])

        # Rendering and showing the image
        draw_connections(displayL, keypoints_with_scores_0, EDGES, 0.4)
        draw_connections(displayR, keypoints_with_scores_1, EDGES, 0.4)
        draw_keypoints(displayL, keypoints_with_scores_0, 0.4)
        draw_keypoints(displayR, keypoints_with_scores_1, 0.4)

        # Optional: resize for display if needed
        displayL = cv2.resize(displayL, (960, 540))
        displayR = cv2.resize(displayR, (960, 540))

        # Combine the frames horizontally
        combined = np.hstack((displayL, displayR))

        # Show in one window
        cv2.imshow("Combined MoveNet Thunder", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()












