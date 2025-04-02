########################################################################
#
# File:   transrot.py
# Author: Matt Zucker
# Date:   January 2012 (Updated April 2021)
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This file shows how to perform image translation and rotation using
# OpenCV, including proper handling of bounding boxes. For another
# example of manipulating transformations, see t_homog.py.

import cv2
import numpy
import sys
import math

WINDOW_NAME = 'Collage'

def main():

    # Check the command line arguments, and inform the user how to run
    # the program if necessary.
    if len(sys.argv) < 5 or (len(sys.argv)-1) % 4:

        print('usage: python', sys.argv[0], 
              'image1 x1 y1 angle1deg image2 x2 y2 angle2deg ...')

        print(' e.g.: python', sys.argv[0], 
              'data/phoenix.jpg 30 20 10 data/swat_logo.png 140 360 -10')

        sys.exit(1)

    # Get the number of pictures
    numpics = (len(sys.argv)-1)//4

    # Initialize an empty array to hold all of the points we will use.
    allpoints = numpy.empty( (0, 1, 2), dtype='float32' )

    # List of all images used.
    images = []

    # List of all transformation matrices used.
    matrices = []

    # List of all outlines of the resulting images after transformation.
    polygons = []

    # Iterate over each argument
    for p in range(numpics):

        # Get filename, x, y, and angle:
        filename = sys.argv[4*p + 1]
        x = float(sys.argv[4*p + 2])
        y = float(sys.argv[4*p + 3])
        a = float(sys.argv[4*p + 4])

        # Read in the image
        image = cv2.imread(filename)

        # Get the width and height
        w = image.shape[1]
        h = image.shape[0]

        # Construct a transformation matrix for the image that achieves
        # the desired rotation and translation
        M = numpy.eye(3,3, dtype='float32')

        # First, construct a matrix to rotate about the center
        M[0:2, :] = cv2.getRotationMatrix2D( (w*0.5, h*0.5), a, 1 )

        # Then consider the translation.
        M[0,2] += x
        M[1,2] += y

        # Construct an array of points on the border of the image.
        # Note this is n-by-1-by-2 as cv2.perspectiveTransform
        # expects.  If you have a more typical n-by-2 array, you can
        # use numpy's reshape method to get it into the correct shape.
        p = numpy.array( [ [[ 0, 0 ]],
                           [[ w, 0 ]],
                           [[ w, h ]],
                           [[ 0, h ]] ], dtype='float32' )

        # Send the points through the transformation matrix.
        pp = cv2.perspectiveTransform(p, M)

        # Add to the lists above
        images.append(image)
        matrices.append(M)
        polygons.append(pp)

        # Append points to the array of all points.
        allpoints = numpy.vstack((allpoints, pp))

    # Compute the bounding rectangle for all points (note this gives
    # integer coordinates).
    box = cv2.boundingRect(allpoints)

    # Get the upper left corner of the rectangle, and its dimensions as well.
    p0 = box[0:2]
    dims = box[2:4]

    # Construct an RGB image with the appropriate size to hold all of the
    # results.
    result = numpy.zeros( (dims[1], dims[0], 3), dtype='uint8' )

    # We need a temporary array to help us compose the images below.
    temp = numpy.empty_like(result)

    # We also need a mask to select the correct pixels.
    mask = numpy.empty( (dims[1], dims[0]), dtype='uint8' )

    # This creates a boolean view of the mask, which we can use to do a
    # masked copy below.  No data is copied.
    bmask = mask.view(bool)

    wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(WINDOW_NAME, wflags)
    cv2.moveWindow(WINDOW_NAME, 50, 50)

    # For each image
    for i in range(len(images)):

        # Get the transformation matrix and update it to put the top
        # left corner at the origin. by pre-multiplying a translation
        # matrix

        M = matrices[i]

        T = numpy.array([[1, 0, -p0[0]],
                         [0, 1, -p0[1]],
                         [0, 0, 1]])

        # use @ for matrix multiplication with numpy arrays (remember
        # * gives element-wise product)
        M = T @ M

        # Update each border polygon to put the top left corner at the
        # origin.
        for j in range(len(polygons[i])):
            polygons[i][j] -= p0

        # Zero out the mask and color in white where the image will go.
        mask[:] = 0
        cv2.fillConvexPoly(mask, polygons[i].astype('int32'), (255,255,255))

        # Warp the image to the destination in the temp image.
        cv2.warpPerspective(images[i], M, tuple(dims), temp)

        cv2.imshow(WINDOW_NAME, temp)
        while cv2.waitKey(5) < 0: pass

        cv2.imshow(WINDOW_NAME, mask)
        while cv2.waitKey(5) < 0: pass

        # Use the mask to copy the pixels into the final image.
        result[bmask] = temp[bmask]

    # Create a window and display the results.
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, result)
    
    while cv2.waitKey(5) < 0: pass

if __name__ == '__main__':
    main()
