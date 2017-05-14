#!/usr/bin/env python

import argparse
import cv2

def print_dimension(img):
	print "image shape: " \
		+ "h=" + str(img.shape[0]) \
		+ ", w=" + str(img.shape[1]) \
		+ ", d=" + str(img.shape[2])

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# 	help = "Path to the image to be scanned")
# args = vars(ap.parse_args())

# load images
img1 = cv2.imread('notre1.jpg',0)           # queryImage
img2 = cv2.imread('notre2.jpg',0) 			# trainImage
# print_dimension(img1)
# print_dimension(img2)

# Initiate SIFT detector
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
	if m.distance < 0.75*n.distance:
		good.append([m])

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

cv2.imshow("Outline", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()