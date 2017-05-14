#!/usr/bin/env python
"""
page-recognition recognize book pages and play a related given sound
Copyright (C) 2017  Ilario Digiacomo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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


# ============== MAIN ===================== #

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# 	help = "Path to the image to be scanned")
# args = vars(ap.parse_args())

# load images
#img1 = cv2.imread('page3_ref.jpg',0)		# queryImage
img2 = cv2.imread('page_unk.jpg',0)			# trainImage
# print_dimension(img1)
# print_dimension(img2)

ratio = float(img2.shape[0])/500
#img1 = resize(img1, ratio) #NOTE: overwriting!
img2 = resize(img2, ratio) #NOTE: overwriting!

FIRST_IMG_INDEX = 1
IMAGE_NUMBER = 3
ref_img = [resize(cv2.imread('page' + str(i) + '_ref.jpg',0),ratio) for i in range(FIRST_IMG_INDEX, IMAGE_NUMBER+FIRST_IMG_INDEX) ]
# count=1
# for img in ref_img:
#     cv2.imshow('Image '+ str(count), img)
#     count+=1

# Initiate ORB detector
orb = cv2.ORB()

# find the keypoints with ORB
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp2, des2 = orb.compute(img2, kp2)

kp_ref = []
des_ref = []
for img in ref_img:
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    kp_ref.append(kp)
    des_ref.append(des)
print "..keypoint and descriptor computed"

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
# matches = bf.match(des1,des2)
# matches = bf.knnMatch(des1,des2, k=2)
match = []
for des in des_ref:
    matches = bf.knnMatch(des,des2, k=2)
    match.append(matches)
print "..matches computed"

# Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#        # Add first matched keypoint to list
#        # if ratio test passes
#        good.append(m)

best_good_match = [0,0]
best_good=None
count=0
for each_match in match:
    good = []
    for m,n in each_match:
        if m.distance < 0.75*n.distance:
           # Add first matched keypoint to list
           # if ratio test passes
           good.append(m)
    l = len(good)
    print ".." + str(l) + " good matches with reference " + str(count)
    if l > best_good_match[0]:
        best_good_match = [l,count]
        best_good = good[:]
    # img3 = drawMatches(ref_img[count], kp_ref[count], img2, kp2, good)
    count+=1
print "\nBEST PAGE: " + str(best_good_match[1]+FIRST_IMG_INDEX) + " with " + str(best_good_match[0]) + " feature match\n"

img3 = drawMatches(ref_img[best_good_match[1]], kp_ref[best_good_match[1]], img2, kp2, best_good) #NOTE: i should use gray images

#cv2.imshow("Outline", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()