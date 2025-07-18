# visualize_rectification.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG: paths to your param file and example images ---
PARAM_FILE   = "params/stereo_params_rectified.npz"
LEFT_EXAMPLE = "img/captures/left_001.png"
RIGHT_EXAMPLE= "img/captures/right_001.png"
OUTPUT_DIR   = "img/output_rectify_vis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- STEP 1: load rectify maps from our .npz ---
data = np.load(PARAM_FILE, allow_pickle=True)
map1_l = data["map1_l"]   # left X map
map2_l = data["map2_l"]   # left Y map
map1_r = data["map1_r"]   # right X map
map2_r = data["map2_r"]   # right Y map

# --- STEP 2: load example stereo pair ---
imgL = cv2.imread(LEFT_EXAMPLE)
imgR = cv2.imread(RIGHT_EXAMPLE)
if imgL is None or imgR is None:
    raise FileNotFoundError("Example images not found!")

# --- STEP 3: apply remap (rectify) ---
rectL = cv2.remap(imgL, map1_l, map2_l, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map1_r, map2_r, cv2.INTER_LINEAR)

# --- STEP 4: visualize before / after ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# original
axs[0,0].imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
axs[0,0].set_title("Original Left")
axs[0,0].axis("off")

axs[0,1].imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
axs[0,1].set_title("Original Right")
axs[0,1].axis("off")

# rectified
axs[1,0].imshow(cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB))
axs[1,0].set_title("Rectified Left")
axs[1,0].axis("off")

axs[1,1].imshow(cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB))
axs[1,1].set_title("Rectified Right")
axs[1,1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rectify_comparison.png"))
plt.show()
