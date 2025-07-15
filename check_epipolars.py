def draw_epipolar_lines(img, interval=40):
    img_lines = img.copy()
    h = img.shape[0]
    for y in range(0, h, interval):
        cv2.line(img_lines, (0, y), (img.shape[1], y), (0, 255, 0), 1)
    return img_lines

# After remapping
rectifiedL_lines = draw_epipolar_lines(rectifiedL)
rectifiedR_lines = draw_epipolar_lines(rectifiedR)

# Combine and show
combined_lines = np.hstack((rectifiedL_lines, rectifiedR_lines))
cv2.imshow("Epipolar Check", combined_lines)
