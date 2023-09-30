import numpy as np
import cv2 as cv
from skimage.feature import match_descriptors, SIFT
from skimage.transform import warp

# Read the two images
img1 = cv.imread("images/img1.jpg", cv.IMREAD_GRAYSCALE)
img5 = cv.imread("images/img5.jpg", cv.IMREAD_GRAYSCALE)

cv.imshow("Image 1", img1)
cv.waitKey(0)
cv.imshow("Image 5", img5)
cv.waitKey(0)

# Detect and compute SIFT features in the images
sift = cv.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints5, descriptors5 = sift.detectAndCompute(img5, None)

# Match SIFT features between the two images
bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors5, k=2)

# Estimate the homography using RANSAC
homography, mask = cv.findHomography(np.array(keypoints1)[matches[:, 0]], np.array(keypoints5)[matches[:, 1]], cv.RANSAC)

# Warp image1 to image2 using the estimated homography
warped_image = warp(img1, homography, output_shape=img5.shape)

# Stitch the two images together
stitched_image = cv.addWeighted(img5, 0.5, warped_image, 0.5, 0)

# Display the stitched image
cv.imshow("Stitched Image", stitched_image)
cv.waitKey(0)
cv.destroyAllWindows()
