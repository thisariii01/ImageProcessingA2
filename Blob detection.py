import cv2 as cv
import numpy as np

# Load the sunflower field image
im = cv.imread('images/the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Define a range of sigma (σ) values for LoG filter
sigma_values = np.arange(1, 10, 0.5)

# Threshold for blob detection
threshold = 0.31

circles = []

# Loop through different sigma values to find the largest circle
for sigma in sigma_values:
    # Apply LoG filter to the grayscale image
    im_log = cv.GaussianBlur(im_gray, (9, 9), sigma, sigma)
    im_log = cv.Laplacian(im_log, cv.CV_64F)
    im_log_abs = np.abs(im_log)
    
    mask = im_log_abs > threshold * im_log_abs.max()

    contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) >= 20:
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius, sigma))

# Find the largest circle
max_radius_circle = max(circles, key=lambda x: x[1])
max_center, max_radius, max_sigma = max_radius_circle

for circle in circles:
    center, radius, _ = circle
    cv.circle(im, center, radius, (0, 255, 0), 1) # Draw the circle on the original image

# Display the result
cv.imshow('Largest Circle', im)
cv.waitKey(0)
cv.destroyAllWindows()

# Report the parameters of the largest circle
print(f"Largest Circle Parameters:")
print(f"Center: {max_center}")
print(f"Radius: {max_radius}")
print(f"Sigma (σ) value used: {max_sigma}")

