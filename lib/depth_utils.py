def calculate_depth(disparity_val, min_disparity=10):

    focal_length_px = 752.9     # px
    baseline_cm = 7.7           # cm
    pixel_size_mm = 1.0         # approx mm per pixel

    if disparity_val < min_disparity:
        disparity_val = min_disparity

    disparity_mm = disparity_val * pixel_size_mm
    depth_mm = (focal_length_px * baseline_cm * 10) / (disparity_mm + 1e-10)
    return depth_mm / 10.0  # cm
