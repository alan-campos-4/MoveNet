import numpy as np
import cv2



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

"""
Coordinates
	#right_eye = keypoints_with_scores[0][0][2]
	#left_elbow = keypoints_with_scores[0][0][7]
	#left_elbow_coords = np.array(left_elbow[:2]*[480,640]).astype(int)
"""

# Draw the edges between the coordinates
def draw_connections(frame, keypoints, edges, confidence_threshold):
	y, x, c = frame.shape
	shaped_array = np.squeeze(np.multiply(keypoints, [y,x,1])) # multiplies the keypoints by the dimesions of the frame
	for edge, color in edges.items():  # for every edge, gets the coordinates of the two points and connects them
		p1, p2 = edge
		y1, x1, c1 = shaped_array[p1]
		y2, x2, c2 = shaped_array[p2]
		if (c1 > confidence_threshold) & (c2 > confidence_threshold):
			cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), edge_color, 2)


