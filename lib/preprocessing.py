
import cv2
import numpy as np

def preprocess(img):
    """
    Resize the image to 256x256 using CUDA and prepare it for MoveNet input.
    """
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(img)
    resized_gpu = cv2.cuda.resize(gpu_mat, (256, 256))
    img_resized = resized_gpu.download()
    img_resized = img_resized.astype(np.int32)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized
