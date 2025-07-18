import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RECTIFY_YAML = "rectify_map_imx219_160deg_1080p.yaml"
IMAGE_SHAPE  = (1080, 1920)  # (height, width) of your images

# --- 1) Load rectify maps from YAML ---
fs = cv2.FileStorage(RECTIFY_YAML, cv2.FILE_STORAGE_READ)
map_l_x = fs.getNode('map_l_x').mat()
map_l_y = fs.getNode('map_l_y').mat()
map_r_x = fs.getNode('map_r_x').mat()
map_r_y = fs.getNode('map_r_y').mat()
fs.release()

# --- 2) Build reference identity maps ---
# identity_x[y,x] = x, identity_y[y,x] = y
identity_y, identity_x = np.mgrid[0:IMAGE_SHAPE[0], 0:IMAGE_SHAPE[1]]

# --- 3) Compute horizontal offsets (dX) ---
dx_l = map_l_x - identity_x
dx_r = map_r_x - identity_x

# plot dX
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(dx_l, cmap='jet')
axs[0].set_title('dX (left)')
axs[1].imshow(dx_r, cmap='jet')
axs[1].set_title('dX (right)')
axs[2].imshow(dx_l - dx_r, cmap='jet')
axs[2].set_title('dX difference (L–R)')
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

# --- 4) Compute vertical offsets (dY) ---
dy_l = map_l_y - identity_y
dy_r = map_r_y - identity_y

# plot dY
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(dy_l, cmap='jet')
axs[0].set_title('dY (left)')
axs[1].imshow(dy_r, cmap='jet')
axs[1].set_title('dY (right)')
axs[2].imshow(dy_l - dy_r, cmap='jet')
axs[2].set_title('dY difference (L–R)')
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

# --- 5) Total displacement magnitude ---
disp_mag_l = np.sqrt(dx_l**2 + dy_l**2)
disp_mag_r = np.sqrt(dx_r**2 + dy_r**2)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(disp_mag_l, cmap='turbo')
axs[0].set_title('Total displacement (left)')
axs[1].imshow(disp_mag_r, cmap='turbo')
axs[1].set_title('Total displacement (right)')
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()
