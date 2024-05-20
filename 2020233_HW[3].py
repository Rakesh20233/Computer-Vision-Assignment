import os

import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from stitching import Stitcher
from stitching.images import Images

stitcher1 = Stitcher(detector="sift", confidence_threshold=0.0)

img1 = cv.imread('panaroma_generation/1.jpg')
img2 = cv.imread('panaroma_generation/2.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img_copy = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.imshow(img_copy)
plt.show()
sift1 = cv.SIFT_create()
kp1 = sift1.detect(gray1, None)

sift2 = cv.SIFT_create()
kp2 = sift2.detect(gray2, None)

img1_updated = cv.drawKeypoints(gray1, kp1, img1)
cv.imwrite('1.jpg', img1)

img2_updated = cv.drawKeypoints(gray2, kp2, img2)
cv.imwrite('2.jpg', img2)

img1_update_1=cv.drawKeypoints(gray1,kp1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('1.jpg',img1_update_1)

img2_updated_2=cv.drawKeypoints(gray2,kp2,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('2.jpg',img2_updated_2)


kp1, des1 = sift1.detectAndCompute(gray1,None)
kp2, des2 = sift2.detectAndCompute(gray2,None)

print("kp1 : ")
print(kp1)
print("des1 : ")
print(des1)
print("kp2 : ")
print(kp2)
print("des2 : ")
print(des2)

# Create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors using BruteForce
matches_bf = bf.match(des1, des2)

# Sort matches based on distance
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

# Draw top 10 matches
image_matches_bf = cv2.drawMatches(img1, kp1, img2, kp2, matches_bf[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display BruteForce matches
cv2.imshow('BruteForce Matches', image_matches_bf)
cv2.waitKey(0)

# Initialize FlannBased matcher
flann = cv2.FlannBasedMatcher()

# Match descriptors using FlannBased
matches_flann = flann.knnMatch(des1, des2, k=2)

# Ratio test to find good matches
good_matches = []
for m, n in matches_flann:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw top 10 good matches
image_matches_flann = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display FlannBased matches
cv2.imshow('FlannBased Matches', image_matches_flann)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute Homography using RANSAC
# homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# All points are in format [cols, rows]
pt_A = [0, 0]
pt_B = [540, 0]
pt_C = [960, 540]
pt_D = [960, 0]

# Here, I have used L2 norm. You can use L1 also.
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))


height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h, w, channel = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv.perspectiveTransform(pts,M)
img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)


# Print the computed Homography matrix
print("Homography Matrix:")
# print(homography)
print(M)


# Assuming you have the computed homography matrix
# homography from the previous step

# Get the dimensions of the images
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]


good_matches1 = []
for m, n in matches_flann:
    if m.distance < 0.9 * n.distance:
        good_matches1.append(m)

src_pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts1 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

print("src_pts1 : ")
print(src_pts1[:4])

print("dst_pts1 : ")
print(dst_pts1[:4])

transform_mat = cv2.getPerspectiveTransform(input_pts, output_pts)

# Warp image2 to align with image1 using the homography matrix
# image2_warped = cv2.warpPerspective(img2, M, (width1 + width2, max(height1, height2)))
img2 = cv.imread('panaroma_generation/2.jpg')
image2_warped = cv2.warpPerspective(img2, M, (maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
img1 = cv.imread('panaroma_generation/1.jpg')
# Combine the original image1 and the warped image2 side-by-side
combined_image = np.concatenate((img1, image2_warped), axis=1)

# Display the original and warped images side-by-side
cv2.imshow('Original and Warped Images', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
img1 = cv.imread('panaroma_generation/1.jpg')
img2 = cv.imread('panaroma_generation/2.jpg')
image2_warped = cv2.warpPerspective(img2, M, (width1 + width2, max(height1, height2)))
combined_image = np.concatenate((img1, image2_warped), axis=1)
stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
cv2.imwrite("2_warped.jpg", image2_warped)
# result = stitcher.stitch(["1.jpg"], ["2_warped.jpg"])

cv.imshow("1", img1)
cv.waitKey(0)
cv.destroyAllWindows()
# img_test = Images.read_image(os.open("panorama_generation/1.jpg"))
# result2 = stitcher1.stitch(['panorama'])
img_test = cv.imread('2_warped.jpg')
# result1 = stitcher1.stitch([img1, img_test])
# Display the stitched panorama without cropping or blending
# cv2.imwrite("test_panorama.jpg", result[1])

print("\nresult : ")
# print(result)
print("\nresult1 : ")
# print(result1)
cv2.imshow('Stitched Panorama (No Cropping or Blending)', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_files = sorted(os.listdir('panaroma_generation'))

# Initialize panorama with the first image
panorama1 = cv2.imread(os.path.join('panaroma_generation', image_files[0]))
#
# # Iterate over pairs of adjacent images
# for i in range(0, len(image_files)-1):
#     # Load the next image
#     next_image = cv2.imread(os.path.join('panaroma_generation', image_files[i]))
#
#     current_image = cv2.imread(os.path.join('panaroma_generation', image_files[i]))
#     next_image = cv2.imread(os.path.join('panaroma_generation', image_files[i + 1]))
#
#     gray3 = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)
#     gray4 = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)
#
#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()
#
#     # Detect keypoints and descriptors
#     keypoints1, descriptors1 = sift.detectAndCompute(gray3, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(gray4, None)
#
#     # Create BFMatcher object
#     bf = cv2.BFMatcher()
#
#     # Match descriptors using BruteForce
#     matches = bf.match(descriptors1, descriptors2)
#
#     # Extract matched keypoints
#     src_pts3 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     dst_pts4 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#
#     # Compute Homography using RANSAC
#     homography1, _ = cv2.findHomography(src_pts3, dst_pts4, cv2.RANSAC, 5.0)
#
#     # Compute homography between current image and previous image
#     # homography = compute_homography(panorama, next_image)
#     # Warp the next image to align with the panorama
#     next_image_warped = cv2.warpPerspective(next_image, homography1,
#                                             (panorama1.shape[1] + next_image.shape[1], max(panorama1.shape[0], next_image.shape[0])))
#
#     # Concatenate the panorama and warped next image
#     panorama1 = np.concatenate((panorama1, next_image_warped), axis=1)
#     cv2.imshow(r'Multi-Stitched Panorama[i]', panorama1)
#
# # Display the final multi-stitched panorama
# cv2.imshow('Multi-Stitched Panorama', panorama1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
import os

# # import the necessary packages
# from imutils import paths
# import numpy as np
# import argparse
# import imutils
# import cv2
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", type=str, required=True,
#                 help="path to input directory of images to stitch")
# ap.add_argument("-o", "--output", type=str, required=True,
#                 help="path to the output image")
# args = vars(ap.parse_args())
#
# print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["panorama_generation"])))
# images = []
# # loop over the image paths, load each one, and add them to our
# # images to stitch list
# for imagePath in imagePaths:
#     image = cv2.imread(imagePath)
#     images.append(image)
#
# # stitching
# print("[INFO] stitching images...")
# stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
# (status, stitched) = stitcher.stitch(images)
#
# # stitching
# if status == 0:
#     # write the output stitched image to disk
#     cv2.imwrite(args["output"], stitched)
#     # display the output stitched image to our screen
#     cv2.imshow("Stitched", stitched)
#     cv2.waitKey(0)
# # otherwise the stitching failed, likely due to not enough keypoints)
# # being detected
# else:
#     print("[INFO] image stitching failed ({})".format(status))

import cv2
image_path = "C:/Users/rakes/Desktop/Computer Vision/Assignment 3/panaroma_generation/"
# image_paths = ["C:/Users/rakes/Desktop/Computer Vision/Assignment 3/panaroma_generation/1.jpg", 'C:/Users/rakes/Desktop/Computer Vision/Assignment 3/panaroma_generation/2.jpg', 'C:/Users/rakes/Desktop/Computer Vision/Assignment 3/panaroma_generation/3.jpg']
# initialized a list of images
imgs = []

# for i in range(len(image_paths)):
#     imgs.append(cv2.imread(image_paths[i]))
    # imgs[i] = cv2.resize(imgs[i], (0, 0), fx=0.4, fy=0.4)
    # this is optional if your input images isn't too large
    # you don't need to scale down the image
    # in my case the input images are of dimensions 3000x1200
    # and due to this the resultant image won't fit the screen
    # scaling down the images
# showing the original pictures
# print(imgs[1])
# if os.path.exists(image_paths[0]):
#     if os.path.isfile(image_paths[0]):
#         cv2.imshow('1', imgs[0])
#         cv2.waitKey(0)
#     else:
#         print("File not exist")
# else:
#     print("Path does not exist")
test = []
for abcd in os.listdir(image_path):
    image_location = os.path.join(image_path, abcd)
    test.append(image_location)
    for i in range(len(test)):
        test1 = cv2.imread(test[i])
        test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2RGB)
        imgs.append(test1)

for i in range(len(imgs)):          ## To see how it is stitching images
    cv2.imshow(f'{i + 1}', imgs[i])
    cv2.waitKey(0)
# cv2.imshow('2', imgs[1])
# cv2.imshow('3', imgs[2])

stitchy = cv2.Stitcher.create()
(dummy, output) = stitchy.stitch(imgs)

stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)

# Stitch the images
status, stitched_image = stitcher.stitch(imgs)
cv2.imshow('final result test', stitched_image)

if dummy != cv2.STITCHER_OK:
    # checking if the stitching procedure is successful
    # .stitch() function returns a true value if stitching is
    # done successfully
    print("stitching ain't successful")
else:
    print('Your Panorama is ready!!!')

# final output
cv2.imshow('final result', output)

cv2.waitKey(0)