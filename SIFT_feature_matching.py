import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def stitching_using_SIFT_features(img1, img2):
    # SIFT Feature Matching
    sift = cv.SIFT_create() 
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    match_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)

    # Compute Homography
    max_iterations = 1000
    inlier_threshold = 5.0
    best_homography = None
    best_inliers = 0

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    for _ in range(max_iterations):
        # Minimum 4 points required to compute homography
        sample_indices = np.random.choice(len(src_pts), 4, replace=False) 
        random_src_pts = np.squeeze(src_pts[sample_indices])
        random_dst_pts = np.squeeze(dst_pts[sample_indices])

        homography, inliers = cv.findHomography(random_src_pts, random_dst_pts, cv.RANSAC, inlier_threshold)
        transformed_pts = np.matmul(homography, np.hstack((random_src_pts, np.ones((4, 1)))).T).T
        distances = np.linalg.norm(transformed_pts[:, :2] / transformed_pts[:, 2, np.newaxis] - random_dst_pts, axis=1)
        inliers = np.sum(distances < inlier_threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_homography = homography

    # Stitching
    warped_img = cv.warpPerspective(img1, best_homography, (img1.shape[1], img1.shape[0]))
    resized_img2 = cv.resize(img2, (warped_img.shape[1], warped_img.shape[0]))
    stitched_img = np.where(warped_img == 0, resized_img2, warped_img)

    return match_img, best_homography, stitched_img

img1 = cv.imread('images/img1.jpg')
img2 = cv.imread('images/img2.jpg')
SIFT_match1, best_homography1, stitched_img1= stitching_using_SIFT_features(img1, img2)

img3 = cv.imread('images/img3.jpg')
SIFT_match2, best_homography2, stitched_img2= stitching_using_SIFT_features(stitched_img1, img3)

img4 = cv.imread('images/img4.jpg')
SIFT_match3, best_homography3, stitched_img3= stitching_using_SIFT_features(stitched_img2, img4)

img5 = cv.imread('images/img5.jpg')
SIFT_match4, best_homography4, stitched_img4= stitching_using_SIFT_features(stitched_img3, img5)

'''fig1 = plt.figure(figsize=(16, 4))
ax1 = fig1.add_subplot(151)
ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
ax1.set_title('Graffiti Image 1')
ax2 = fig1.add_subplot(152)
ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
ax2.set_title('Graffiti Image 2')
ax3 = fig1.add_subplot(153)
ax3.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
ax3.set_title('Graffiti Image 3')
ax4 = fig1.add_subplot(154)
ax4.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
ax4.set_title('Graffiti Image 4')
ax5 = fig1.add_subplot(155)
ax5.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
ax5.set_title('Graffiti Image 5')
plt.show()'''

'''fig2 = plt.figure(figsize=(8, 16))
ax1 = fig2.add_subplot(421) 
ax1.imshow(cv.cvtColor(SIFT_match1, cv.COLOR_BGR2RGB))
ax1.set_title('SIFT Feature Matching with img1.ppm and img2.ppm')
ax2 = fig2.add_subplot(422)
ax2.imshow(cv.cvtColor(stitched_img1, cv.COLOR_BGR2RGB))
ax2.set_title('Stitched Image with img1.ppm and img2.ppm')
ax3 = fig2.add_subplot(423)
ax3.imshow(cv.cvtColor(SIFT_match2, cv.COLOR_BGR2RGB))
ax3.set_title('SIFT Feature Matching with stitched image1 and img3.ppm')
ax4 = fig2.add_subplot(424)
ax4.imshow(cv.cvtColor(stitched_img2, cv.COLOR_BGR2RGB))
ax4.set_title('Stitched Image with stitched image1 and img3.ppm')
ax5 = fig2.add_subplot(425)
ax5.imshow(cv.cvtColor(SIFT_match3, cv.COLOR_BGR2RGB))
ax5.set_title('SIFT Feature Matching with stitched image2 and img4.ppm')
ax6 = fig2.add_subplot(426)
ax6.imshow(cv.cvtColor(stitched_img3, cv.COLOR_BGR2RGB))
ax6.set_title('Stitched Image with stitched image2 and img4.ppm')
ax7 = fig2.add_subplot(427)
ax7.imshow(cv.cvtColor(SIFT_match4, cv.COLOR_BGR2RGB))
ax7.set_title('SIFT Feature Matching with stitched image3 and img5.ppm')
ax8 = fig2.add_subplot(428)
ax8.imshow(cv.cvtColor(stitched_img4, cv.COLOR_BGR2RGB))
ax8.set_title('Stitched Image with stitched image3 and img5.ppm')
plt.show()'''

cv.imshow('Stitched Image', stitched_img4)
cv.waitKey(0)
cv.destroyAllWindows()
print('Homography Matrix 1:\n', best_homography4)
