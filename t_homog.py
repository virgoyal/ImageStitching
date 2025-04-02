########################################################################
#
# File:   t_homog.py
# Author: Matt Zucker
# Date:   February 2015 (Updated April 2021)
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This file shows how to compose translations and homographies nicely.

import numpy as np
import cv2

WINDOW_NAME = 'Homography + translation demo'

def labelAndWaitForKey(image, text):

    # Get the image height - the first element of its shape tuple.
    h = image.shape[0]
    
    # Note that even though shapes are represented as (h, w), pixel
    # coordinates below are represented as (x, y). Confusing!
    cv2.putText(image, text, (16, h-16), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,0,0), 3, cv2.LINE_AA)

    cv2.putText(image, text, (16, h-16), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, image)

    # We could just call cv2.waitKey() instead of using a while loop
    # here, however, on some platforms, cv2.waitKey() doesn't let
    # Ctrl+C interrupt programs. This is a workaround.
    while cv2.waitKey(15) < 0: pass

def main():

    # Load an image
    orig = cv2.imread('data/swat_logo.png')

    # Get its size 
    h, w = orig.shape[:2]
    size = (w, h)

    wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE
    
    cv2.namedWindow(WINDOW_NAME, wflags)
    cv2.moveWindow(WINDOW_NAME, 50, 50)

    # Show it
    labelAndWaitForKey(orig.copy(), 'orig')

    ######################################################################
    # Now make a neat "keystone" type homography by composing a
    # translation, a simple homography, and the inverse translation.

    # Translate center of image to (0,0)
    Tfwd = np.eye(3)
    Tfwd[0,2] = -0.5 * w
    Tfwd[1,2] = -0.5 * h

    # Get inverse of that
    Tinv = np.linalg.inv(Tfwd)

    # Homography that decreases homogeneous "w" coordinate with increasing
    # depth so bottom rows appear "closer" than top rows".
    H = np.eye(3)
    H[2,1] = -0.002

    S = np.eye(3)
    S[0,0] = 0.5
    S[1,1] = 0.5

    # Compose the three transforms together using matrix
    # multiplication. 
    #
    # Use @ operator to do matrix multiplication on numpy arrays
    # (remember * gives element-wise product)

    H = S @ Tinv @ H @ Tfwd

    # Show the warped version. Note that you don't need to pass a
    # "destination" image into warpPerspective - you can just get the
    # return value.
    warped = cv2.warpPerspective(orig, H, size)

    # Show it
    labelAndWaitForKey(warped, 'warped')

    ######################################################################
    # Now translate the final warped image so we can see it all. This uses
    # the same trick from transrot.py, except instead of modifying the
    # homography matrix directly, just composes it with a translation.

    # Get corner points of original image - note this is shaped as an
    # n-by-1-by-2 array, because that's what cv2.perspectiveTransform
    # expects. If you have a more typical n-by-2 array, you can use
    # numpy's reshape method to get it into the correct shape.
    p = np.array( [ [[0, 0]],
                    [[w, 0]],
                    [[w, h]],
                    [[0, h]] ], dtype='float32' )

    # Map through warp
    pp = cv2.perspectiveTransform(p, H)

    # Get integer bounding box of form (x0, y0, width, height)
    box = cv2.boundingRect(pp)

    # Separate into dimensions and origin
    dims = box[2:4]
    p0 = box[0:2]

    # Create translation transformation to shift image
    Tnice = np.eye(3)
    Tnice[0,2] -= p0[0]
    Tnice[1,2] -= p0[1]

    # Compose them via matrix multiplication
    Hnice = Tnice @ H

    # Show it
    warpedNice = cv2.warpPerspective(orig, Hnice, dims)
    labelAndWaitForKey(warpedNice, 'warped nice')

if __name__ == '__main__':
    main()
