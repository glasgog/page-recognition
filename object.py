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
from random import randint  # for features color randomization

import lib.vlc as vlc


def print_dimension(img):
    print "image shape: " \
        + "h=" + str(img.shape[0]) \
        + ", w=" + str(img.shape[1]) \
        + ", d=" + str(img.shape[2])


def resize(img, ratio):
    """ height is the reference 
    ratio have to be float """
    dimension = (int(img.shape[1] / ratio), int(img.shape[0] / ratio))  # (w,h)
    print "resizing at: " + str(dimension)
    print " with ratio: " + str(ratio)
    resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
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

    DEBUG = False

    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((rows2, cols2, 3), dtype='uint8')
    # fill the output with the second image
    out = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    if not matches == None:
        img1 = img1.copy()  # Funziona?
        ratio = float(4 * img1.shape[0]) / img2.shape[0]
        if DEBUG:
            print "ratio=" + str(ratio)

        img1 = resize(img1, ratio)  # WARNING: sto sovrascrivendo immagine

        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]

        DELTAX = DELTAY = rows2 / 16  # intero. Spostamento rispetto angolo superiore destro

        # Place the resized first image to on the second with a delta from the
        # right corner
        out[DELTAY:rows1 + DELTAY, cols2 - cols1 -
            DELTAX:cols2 - DELTAX] = np.dstack([img1, img1, img1])

        for mat in matches:
            if not mat == None:
                # Get the matching keypoints for each of the images
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx

                # x - columns, y - rows
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

                # ridimensiono le coord dei punti dato che ho ridimensionato le
                # ref img
                (x1, y1) = (int(float(x1) / ratio), int(float(y1) / ratio))
                # aggiungo l'offset dovuto alla posizione di visualizzazione
                (x1, y1) = (x1 + cols2 - cols1 - DELTAX, y1 + DELTAY)

                # Draw a small circle at both co-ordinates
                radius = 4
                color = (randint(0, 255), randint(0, 128),
                         randint(0, 128))  # random color
                thickness = 1
                cv2.circle(out, (int(x1), int(y1)), radius, color, thickness)
                cv2.circle(out, (int(x2), int(y2)), radius, color, thickness)

                # Draw a line in between the two points
                cv2.line(out, (int(x1), int(y1)),
                         (int(x2), int(y2)), color, thickness)

    # Show the image
    cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return True


# ============== MAIN ===================== #

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# 	help = "Path to the image to be scanned")
# args = vars(ap.parse_args())

# load images
# img1 = cv2.imread('page3_ref.jpg',0)		# queryImage
# img2 = cv2.imread('page_unk.jpg',0)		# trainImage
# print_dimension(img1)
# print_dimension(img2)

# ratio = float(img2.shape[0])/500
# img1 = resize(img1, ratio) #NOTE: overwriting!
# img2 = resize(img2, ratio) #NOTE: overwriting!

cv2.namedWindow('Matched Features', cv2.CV_WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)
# ret = cap.set(3,640)
# ret = cap.set(4,480

FIRST_IMG_INDEX = 1
IMAGE_NUMBER = 3
# lista delle immagini di riferimento: ["page1_ref.jpg", "page2_ref.jpg", "page3_ref.jpg"]
# devono essere gia' ridimensionate
ref_img = [cv2.imread('page' + str(i) + '_ref.jpg', 0)
           for i in range(FIRST_IMG_INDEX, IMAGE_NUMBER + FIRST_IMG_INDEX)]

# initialize sound file
# sound_file = "sounds/1.mp3"
# p = vlc.MediaPlayer(sound_file)
sound_files = ["sounds/" + str(i) + ".mp3" for i in range(
    FIRST_IMG_INDEX, IMAGE_NUMBER + FIRST_IMG_INDEX)]
sound = []
for name in sound_files:
    sound.append(vlc.MediaPlayer(name))

while(True):
    res, frame = cap.read()
    if not res:
        continue  # se non viene letto il frame passo al ciclo successivo

    print "original frame shape: " + str(frame.shape)
    # ratio=float(frame.shape[0])/500
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    img2 = resize(gray, 1)
    print "resized frame shape: " + str(img2.shape)

    # Initiate ORB detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp2 = orb.detect(img2, None)
    # compute the descriptors with ORB
    kp2, des2 = orb.compute(img2, kp2)

    kp_ref = []
    des_ref = []
    for img in ref_img:
        kp = orb.detect(img, None)
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
        if not des2 == None:  # nell'immagine da esaminare devono esserci delle feature riconoscibili
            matches = bf.knnMatch(des, des2, k=2)
        else:
            matches = None
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

    best_good_match = [0, 0]  # [number_of_matches,index]
    best_good = None
    count = 0
    for each_match in match:
        # type(each_match)
        if each_match == None:
            continue
        good = []
        for m, n in each_match:
            if m.distance < 0.75 * n.distance:
                # Add first matched keypoint to list
                # if ratio test passes
                good.append(m)
        l = len(good)
        print ".." + str(l) + " good matches with reference " + str(count)
        if l > best_good_match[0]:
            best_good_match = [l, count]
            best_good = good[:]
        # img3 = drawMatches(ref_img[count], kp_ref[count], img2, kp2, good)
        count += 1
    print "\nBEST PAGE: " + str(best_good_match[1] + FIRST_IMG_INDEX) + " with " + str(best_good_match[0]) + " feature match\n"

    # per non bloccare il video, se non c'e' match disegno solo il frame
    if best_good_match[0] >= 10:
        res = drawMatches(ref_img[best_good_match[1]], kp_ref[best_good_match[
                          1]], img2, kp2, best_good)  # NOTE: i should use gray images
        if not sound[best_good_match[1]].is_playing():
            # p.play()
            sound[best_good_match[1]].play()
    else:
        res = drawMatches(np.zeros((1, 1, 1), dtype='uint8'),
                          None, img2, None, None)
        # p.stop()
        for s in sound:
            s.stop()
    if not res:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
