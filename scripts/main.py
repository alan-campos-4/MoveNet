import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off different numerical values due to rounding errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # enables more tf instructions in operations
sys.path.insert(0, '/home/jetson_0/Documents/MoveNet/lib')
from pipeline import gstreamer_pipeline
import cv2
import numpy as np
import tensorflow as tf




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
    # Coordinates
    #right_eye = keypoints_with_scores[0][0][2]
    #left_elbow = keypoints_with_scores[0][0][7]
    #left_elbow_coords = np.array(left_elbow[:2]*[480,640]).astype(int)


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
    interpreter = tf.lite.Interpreter(model_path='models/movenet-thunder.tflite')
    interpreter.allocate_tensors()

    # Reads the camera and captures the video.
    cap = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

    while True:
        if (cap.isOpened()==False):
            print("Error: couldn't open the camera.")
            break

        ret, frame = cap.read()

        if not ret:
            print("Error: can't receive frame.")
            break

        # Reshape image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        #print(keypoints_with_scores)

        # Rendering and showing the image
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)   

        # Display the result 
        cv2.imshow('Movenet Thunder', frame)

        # Break the loop if the 'Q' key is pressed
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
