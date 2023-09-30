import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def image_superimposing(bg_img, fg_img, flag_blending, alpha=0.4, n1 = 1.5, n2 = 2):
    # Define four points on the background image 
    background_points = np.array(flag_blending, dtype=np.float32)

    # Define the four corresponding points on the superimpose image
    superimpose_points = np.array([[0, 0], [fg_img.shape[1], 0], [0, fg_img.shape[0]], [fg_img.shape[1], fg_img.shape[0]]], dtype=np.float32)

    homography_matrix, _ = cv.findHomography(superimpose_points, background_points)
    result = cv.warpPerspective(fg_img, homography_matrix, (bg_img.shape[1], bg_img.shape[0]))
    blended_image = cv.addWeighted(bg_img, n1 * (1 - alpha), result, n2 * alpha, 0)

    return blended_image

# Load image set 1
bg_img1 = cv.imread('images/background.jpg')
fg_img1 = cv.imread('images/flag.png')
blending1 = [[30, 40], [100, 55], [27, 105], [100, 107]]
si_img1 = image_superimposing(bg_img1, fg_img1, blending1, 0.3) # Adjust the alpha value for blending

# Load image set 2
bg_img2 = cv.imread('images/independenceSquare.jpeg')
fg_img2 = cv.imread('images/slflag.webp')
blending2 = [[180, 275], [440, 235], [180, 390], [440, 370]]
si_img2 = image_superimposing(bg_img2, fg_img2, blending2, 0.4) # Adjust the alpha value for blending

# Load image set 3
bg_img3 = cv.imread('images/goose.jpg')
fg_img3 = cv.imread('images/horse.png')
blending3 = [[186, 8], [240, 0], [189, 81], [247, 71]]
si_img3 = image_superimposing(bg_img3, fg_img3, blending3, 0.5) # Adjust the alpha value for blending

# Load image set 4
bg_img4 = cv.imread('images/phone.jpeg')
fg_img4 = cv.imread('images/fb.jpeg')
blending4 = [[670, 338], [723, 343], [670, 420], [723, 425]]
si_img4 = image_superimposing(bg_img4, fg_img4, blending4, 0.45) # Adjust the alpha value for blending

# Load image set 5
bg_img5 = cv.imread('images/sky.jpg')
fg_img5 = cv.imread('images/woman.jpeg')
blending5 = [[1, 1], [719, 1], [1, 410], [719, 410]]
si_img5 = image_superimposing(bg_img5, fg_img5, blending5, 0.5, 1, 1) # Adjust the alpha, n1, n2 values for blending

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(321)
ax1.imshow(cv.cvtColor(si_img1, cv.COLOR_BGR2RGB))
# ax1.set_title('Superimposed Image 1')
ax2 = fig.add_subplot(322)
ax2.imshow(cv.cvtColor(si_img2, cv.COLOR_BGR2RGB))
# ax2.set_title('Superimposed Image 2')
ax3 = fig.add_subplot(323)
ax3.imshow(cv.cvtColor(si_img3, cv.COLOR_BGR2RGB))
# ax3.set_title('Superimposed Image 3')
ax4 = fig.add_subplot(324)
ax4.imshow(cv.cvtColor(si_img4, cv.COLOR_BGR2RGB))
# ax4.set_title('Superimposed Image 4')
ax5 = fig.add_subplot(325)
ax5.imshow(cv.cvtColor(si_img5, cv.COLOR_BGR2RGB))
# ax5.set_title('Superimposed Image 5')
plt.show()
