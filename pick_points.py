########################################################################
#
# File:   pick_points.py
# Author: Matt Zucker
# Date:   January 2019 (updated April 2021)
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This file is a convenient helper program to let you pick an ordered
# list of points in an image. You DON'T need to fully understand this
# program to do project 4 but you might find it interesting.

import cv2
import numpy as np
import sys, os

# Max dims of image on screen, change these to get bigger/smaller window
MAX_WIDTH = 1024
MAX_HEIGHT = 600

# Factors for zooming in/out
ZOOM_IN_FACTOR = np.sqrt(2)
ZOOM_OUT_FACTOR = np.sqrt(2)/2

# Click radius for points on screen
MIN_DIST_PX = 24

######################################################################
# Create a class to help with point picking. 

class Picker(object):
    
    # Initialize with an input image and a destination filename
    def __init__(self, image, text_filename):

        # Save a copy of the original image
        self.orig_image = image

        # Stash dimensions of original image
        h, w = image.shape[:2]
        self.orig_image_size = (w, h)

        # Stash the filename where we will save points
        self.text_filename = text_filename

        # See if we can load points from file
        try:
            self.orig_points = np.genfromtxt(self.text_filename, dtype=float)
            if not len(self.orig_points):
                self.orig_points = np.empty((0, 2))
                self.cur_index = -1
            else:
                self.cur_index = 0
        except:
            self.orig_points = np.empty((0, 2))
            self.cur_index = -1

        # Determine minimum scaling factor of image 
        fy = MAX_HEIGHT / float(h)
        fx = MAX_WIDTH / float(w)

        self.zoom = min(1.0, min(fx, fy))
        self.min_zoom = self.zoom
        
        # Set display size to scaled dimensions
        self.display_size = (int(round(w*self.zoom)), int(round(h*self.zoom)))

        # Initialize current scroll offset to 0, 0
        self.scroll_offset = np.array([0., 0.])
        
        # Create a window with OpenCV
        self.window = 'Pick points to save to ' + text_filename
        wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE

        cv2.namedWindow(self.window, wflags)
        cv2.moveWindow(self.window, 50, 50)

        # Transform the image to get ready for display
        self.transform_image()

        # Set up some internal flags for mouse interactions
        self.is_modified = False
        self.dragging = False
        self.mouse_active = True

        # Tell OpenCV we want mouse events
        cv2.setMouseCallback(self.window, self.mouse, self)

    # Create an affine transformation to display part or all of the
    # original image, then warp the image through the transform.
    def transform_image(self):

        # Get some short variable names to make a concise matrix definition
        z = self.zoom
        sx, sy = self.scroll_offset
        dw, dh = self.display_size
        ow, oh = self.orig_image_size

        # Define the forward transformation as a 3x3 matrix
        # in homogeneous coordinates (homography)
        self.M_forward = np.array([
            [z, 0, z*sx + 0.5*(dw - z*ow)],
            [0, z, z*sy + 0.5*(dh - z*oh)],
            [0, 0, 1]])

        # Define the inverse transformation as a 3x3 matrix
        self.M_inverse = np.linalg.inv(self.M_forward)

        # If zooming in, use box filter; otherwise use nearest neighbor.
        if self.zoom < 1:
            flags = cv2.INTER_AREA
        else:
            flags = cv2.INTER_NEAREST

        # Rescale and translate the image itself
        self.transformed_image = cv2.warpAffine(
            self.orig_image, self.M_forward[:2],
            self.display_size, flags=flags)

        # Project the points and prepare the display image
        self.project_points()

    # Given a transformed image, project the current set of points
    # and display it onto the image.
    def project_points(self):

        # Make a display image by copying the transformed image
        self.display_image = self.transformed_image.copy()

        # If no points, we are done!
        if not len(self.orig_points):
            self.display_points = self.orig_points.copy()
            return

        # The cv2.perspectiveTransform function needs an n-by-1-by-2
        # array of input points.
        src = self.orig_points.reshape(-1, 1, 2)

        # Project the points through the homography
        dst = cv2.perspectiveTransform(src, self.M_forward)

        # Convert to int (so we can conveniently display points) and
        # reshape back to n-by-2
        self.display_points = np.round(dst).astype(int).reshape(-1, 2)

        # Now draw points with labels on the screen
        blk = (0, 0, 0)
        
        # For each point
        for i, p in enumerate(self.display_points):

            # If current point, cyan else purple
            if i == self.cur_index:
                fg = (255, 255, 0)
            else:
                fg = (255, 0, 255)

            # 2 iterations: first black background then bright foreground
            for crad, color, trad in [(4, blk, 2), (3, fg, 1)]:

                cv2.circle(self.display_image, tuple(p),
                           crad, color, -1, cv2.LINE_AA)
                
                cv2.putText(self.display_image, str(i+1),
                            tuple(p + (5, 5)), cv2.FONT_HERSHEY_PLAIN,
                            1.0, color, trad, cv2.LINE_AA)

    # Given a point in the window coordinate frame, search for
    # index of the closest point within MIN_DIST_PX or return -1
    def closestPoint(self, p_window):

        # If no points, return -1
        if not len(self.orig_points):
            return -1

        # Compute pointwise deltas
        deltas = self.display_points - p_window

        # Get squared distances
        dists2 = (deltas**2).sum(axis=1)

        # Get index of minimum squared distance
        idx = dists2.argmin()

        # If within minimum distance, return index, otherwise -1
        if np.sqrt(dists2[idx]) < MIN_DIST_PX:
            return idx
        else:
            return -1

    # OpenCV mouse callback for window.
    def mouse(self, event, x, y, flag, param):

        # Do nothing if mouse ignored.
        if not self.mouse_active:
            return

        # Point in window coords
        p_window = np.array((x, y))

        # Map through inverse of homography to get image coords
        src = p_window.astype(float).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.M_inverse)
        p_image = dst.reshape(-1, 2)

        # Was left mouse button depressed?
        if event == cv2.EVENT_LBUTTONDOWN:

            # Get the index of the closest point within range of click location
            idx = self.closestPoint(p_window)

            # No closest point found?
            if idx == -1:

                # Append a new point onto the array of original points.
                self.orig_points = np.vstack( ( self.orig_points,
                                                p_image.reshape(1, 2) ) )

                # Make the current index be the index of the new point
                self.cur_index = len(self.orig_points)-1

                # Modify mouse flags
                self.dragging = True
                self.mouse_point = p_window
                self.is_modified = True
                
                # Update the projected points for display
                self.project_points()
                
            else:

                # There was a closest point found.
                
                # Modify mouse flags.
                self.cur_index = idx
                self.dragging = True
                self.mouse_point = p_window

                # Update the display to reflect change of current point.
                self.project_points()

        elif self.dragging and event == cv2.EVENT_MOUSEMOVE:

            # Move the current point by the mouse displacement
            mouse_displacement = (p_window - self.mouse_point) / self.zoom
            self.orig_points[self.cur_index] += mouse_displacement

            # Update flags
            self.mouse_point = p_window
            self.is_modified = True

            # Update the display
            self.project_points()
            
        elif self.dragging and event == cv2.EVENT_LBUTTONUP:

            # Left mouse button up, we are no longer dragging.
            self.dragging = False

    # Set the current point one forward or back (depending on direction)
    def increment_current_point(self, direction):

        # If no points, do nothing much
        if not len(self.orig_points):
            return

        # Handle wrap-around and generate new index
        self.cur_index = (self.cur_index + direction) % len(self.orig_points)

        # Update the display
        self.project_points()
        
        # See if we need to scroll the screen
        dx, dy = self.display_points[self.cur_index]

        onscreen = (dx >= 0 and dx <= self.display_size[0] and
                    dy >= 0 and dy <= self.display_size[1])

        # Scroll if necessary.
        if not onscreen:
            self.center_current_point() 

    # Reorder the current point one forward or back (depending on direction)
    def reorder_current_point(self, direction):

        # If no points, do nothing much
        if not len(self.orig_points):
            return

        new_idx = (self.cur_index + direction) % len(self.orig_points)
        tmp = self.orig_points[new_idx].copy()
        self.orig_points[new_idx] = self.orig_points[self.cur_index]
        self.orig_points[self.cur_index] = tmp

        self.cur_index = new_idx

        self.is_modified = True

        self.project_points()

    # Reset the view
    def reset_view(self):
        self.scroll_offset = np.array([0., 0.])
        self.zoom = self.min_zoom
        self.transform_image()

    # Set the scroll offset to put the current point in the center of
    # the window.
    def center_current_point(self):

        if self.cur_index < 0 or not len(self.orig_points):
            return

        p = self.orig_points[self.cur_index]

        self.scroll_offset = 0.5*np.array(self.orig_image_size) - p

        self.transform_image()

    # Show a bunch of strings on some grayed-out text.
    def show_strings(self, strlist):

        # Darken image by scaling down to 25% intensity
        dimmed_image = self.display_image//4

        # For each line in the list
        for i, line in enumerate(strlist):

            # For each pair of strings in the line
            for j, text in enumerate(line):

                # Display the text in white.
                cv2.putText(dimmed_image, text, (20+150*j, i*20+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                        

        # Show the dimmed image
        cv2.imshow(self.window, dimmed_image)

    # Show the help screen
    def help(self):

        # List of pairs of strings for display
        helpstr = [
            ('Click to add points, or drag points with mouse.', ''),
            ('', ''),
            ('Keys:', ''),
            ('', ''),
            ('+ or -','zoom image'),
            ('wasd or WASD','scroll image'),
            ('[ or ]','jump to previous/next point'),
            ('{ or }','reorder current point'),
            ('c','center on current point'),
            ('p', 'delete current point'),
            ('SPACE','reset view'),
            ('ESC','exit'),
            ('',''),
            ('?', 'show this help screen'),
            ('',''),
            ('Press any key to dismiss this help screen.','')
        ]

        # Disable mouse 
        self.mouse_active = False

        # Show text
        self.show_strings(helpstr)

        # Wait for any key
        while cv2.waitKey(5) < 0:
            pass

        # Re-enable mouse
        self.mouse_active = True

    # Show the screen to prompt save before quitting
    def prompt_quit(self):

        # Quit immediately if no points modified.
        if not self.is_modified:
            return True

        # Compose text to show
        
        prompt = 'Do you want to save {} points to {}?'.format(
            len(self.orig_points), self.text_filename)
        
        strings = [(prompt, ''),
                   ('', ''),
                   ('Y', 'Yes, save to ' + self.text_filename),
                   ('N', 'No, quit without saving'),
                   ('', ''),
                   ('Any other key resumes interactive mode.', '')]

        # Disable mouse
        self.mouse_active = False

        # Show text
        self.show_strings(strings)

        # Wait for any key
        while True:
            key = cv2.waitKey(5)
            if key >= 0:
                break

        # See what key was
        c = chr(key & 0xff).lower()

        if c == 'y':
            print('Saving to {}'.format(self.text_filename))
            np.savetxt(self.text_filename, self.orig_points, fmt='%.15G')
            done = True
        elif c == 'n':
            print('Quitting without saving!')
            done = True
        else:
            done = False

        # Re-enable mouse
        self.mouse_active = True

        # Return True if time to quit
        return done

    # Get rid of the current point
    def delete_current_point(self):
        
        if self.cur_index < 0 or not len(self.orig_points):
            return
        
        self.orig_points = np.vstack( (
            self.orig_points[:self.cur_index],
            self.orig_points[self.cur_index+1:] ) )
        
        if self.cur_index >= len(self.orig_points):
            self.cur_index = len(self.orig_points)-1
            
        self.is_modified = True
        
        self.project_points()

    # Relative scrolling of window
    def scroll_window(self, sx, sy):
        self.scroll_offset += (sx, sy)
        self.transform_image()

    # Zoom image
    def zoom_image(self, factor):
        self.zoom = min(4.0, max(self.min_zoom, self.zoom * factor))
        self.transform_image()

    # The main loop for this object
    def run(self):

        # Start by showing help screen
        self.help()

        # Forever loop
        while True:

            # Show display image
            cv2.imshow(self.window, self.display_image)

            # Get the key
            key = cv2.waitKey(5)

            if key < 0:
                # No key pressed, just loop
                continue

            # Get the ASCII character from the integer keycode
            c = chr(key & 0xff)

            # Figure out how much to scroll based on zoom
            scroll_amount = 32.0*self.min_zoom / self.zoom

            # Scroll more if CAPS
            if c.isupper():
                c = c.lower()
                scroll_amount *= 8

            # Handle keys
            if c in '+=':
                self.zoom_image(ZOOM_IN_FACTOR)
            elif c == '-':
                self.zoom_image(ZOOM_OUT_FACTOR)
            elif c == 'a':
                self.scroll_window(scroll_amount, 0)
            elif c == 'd':
                self.scroll_window(-scroll_amount, 0)
            elif c == 'w':
                self.scroll_window(0, scroll_amount)
            elif c == 's':
                self.scroll_window(0, -scroll_amount)
            elif c == 'c':
                self.center_current_point()
            elif c == 'p':
                self.delete_current_point()
            elif c == '[':
                self.increment_current_point(-1)
            elif c == ']':
                self.increment_current_point(1)
            elif c == '{':
                self.reorder_current_point(-1)
            elif c == '}':
                self.reorder_current_point(1)
            elif c == ' ':
                self.reset_view()
            elif c == '?':
                self.help()
            elif key == 27:
                if self.prompt_quit():
                    break


######################################################################
# Our main function

def main():

    # Collect command-line args
    if len(sys.argv) != 2:
        print('usage: python', sys.argv[0], 'IMAGEFILE')
        sys.exit(1)

    image_file = sys.argv[1]

    # Load image file
    orig = cv2.imread(image_file)

    if orig is None:
        print('image not found:', image_file)
        sys.exit(1)

    base, ext = os.path.splitext(image_file)
    text_filename = base + '.txt'

    # Create a Picker object and run it
    p = Picker(orig, text_filename)
    p.run()

if __name__ == '__main__':
    main()

    
