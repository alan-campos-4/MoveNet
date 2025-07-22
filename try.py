# Apply slight Gaussian blur to reduce high-frequency noise
# arr_l_rect = cv2.GaussianBlur(arr_l_rect, (5, 5), 0)
# arr_r_rect = cv2.GaussianBlur(arr_r_rect, (5, 5), 0)

# Apply bilateral filter to reduce noise while preserving edges
arr_l_rect = cv2.bilateralFilter(arr_l_rect, 9, 75, 75)
arr_r_rect = cv2.bilateralFilter(arr_r_rect, 9, 75, 75)


disp_arr = disparity_8bpp.cpu() #After this row, add the code below

# Apply median blur to reduce salt-and-pepper noise in disparity map
disp_arr = cv2.medianBlur(disp_arr, 5)
