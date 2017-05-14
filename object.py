#!/usr/bin/env python

import numpy as np
import argparse
import cv2

def print_dimension(img):
	print "image shape: " \
		+ "h=" + str(img.shape[0]) \
		+ ", w=" + str(img.shape[1]) \
		+ ", d=" + str(img.shape[2])

def resize(img, ratio):
	""" height is the reference 
        ratio have to be float """
	dimension=(int(img.shape[1]/ratio),int(img.shape[0]/ratio)) #(w,h)
	print "resizing at: " + str(dimension)
	print " with ratio: " + str(ratio)
	resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
	return resized

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    OpenCV 2.4.9 doesn't support cv2.drawMatchesKnn().
    This is a function with similar use, that return
    a montage of two images with they respective keypoints
    connected by lines.

    img1,img2 - Grayscale images
    kp1,kp2 - List of keypoints
    matches - List of matches of corresponding keypoints

    Any OpenCV keypoint matching algorithm could be used
    for getting kp and matches 
    """

    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


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

ratio = float(img2.shape[0])/500
img1 = resize(img1, ratio) #NOTE: overwriting!
img2 = resize(img2, ratio) #NOTE: overwriting!

# Initiate ORB detector
orb = cv2.ORB()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
       # Add first matched keypoint to list
       # if ratio test passes
       good.append(m)

img3 = drawMatches(img1, kp1, img2, kp2, good) #NOTE: i should use gray images

#cv2.imshow("Outline", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()