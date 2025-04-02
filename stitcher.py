# TODO: Joey, Vir

import sys, os
import numpy as np
import cv2

def main():

    # load all data from command line
    #
    # note: the program currently supports specifying more than two
    # images on the command line but you only need to support 1 max
    # in your project

    if len(sys.argv) < 3:
        print('usage: python stitcher.py IMAGE1 IMAGE2')
        sys.exit(1)

    dataset = []
    num_points = None

    for image_filename in sys.argv[1:]:

        image = cv2.imread(image_filename)

        if image is None: 
            print('error loading', image)
            sys.exit(1)

        h, w = image.shape[:2]
        print('loaded {}x{} image from {}'.format(h, w, image_filename))

        basename, _ = os.path.splitext(image_filename)

        points_filename = basename + '.txt'

        try:
            points = np.genfromtxt(points_filename)
        except OSError:
            print('error reading', points_filename)
            sys.exit(1)

        assert len(points.shape) == 2 and points.shape[1] == 2

        if num_points is None:
            num_points = len(points)
        else:
            assert num_points == len(points)

        # turn into shape (n, 1, 2) that findHomography expects
        points = points.reshape(num_points, 1, 2)

        # make it float32
        points = points.astype(np.float32)

        print('read {} points from {}'.format(len(points), points_filename))
        print()

        dataset.append((image, points))

    # dataset now contains [(imageA, pointsA), (imageB, pointsB)]

    print('ready to stitch images together into output.jpg')

    # TODO: generate output.jpg!
    print(f"# of images in dataset = {len(dataset)} ")
    ref_img, ref_points = dataset[0] # first img is considered to be the reference img
    ref_h, ref_w =  ref_img.shape[:2]
    ref_corners = np.array([[[0,0]],[[ref_w,0]],[[ref_w,ref_h]],[[0,ref_h]]], dtype=np.float32)

    homographies= [np.eye(3,dtype=np.float32)] # identity matrix for first image
    all_corners =[]
    all_corners.append(ref_corners)

    
    for index in range(1,len(dataset)):
        img,pts = dataset[index]

        H, mask = cv2.findHomography(pts, ref_points)
        homographies.append(H)

        h,w = img.shape[:2]
        corners = np.array([[[0,0]],[[w,0]],[[w,h]],[[0,h]]], dtype=np.float32)
        transf_corners = cv2.perspectiveTransform(corners,H)
        all_corners.append(transf_corners)

    # print(all_corners)
    all_corners_arr =np.vstack(all_corners)
    x0,y0,wc,hc = cv2.boundingRect(all_corners_arr)
    T = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]], dtype=np.float32)
   
    warped_list = []
    for index in range(len(dataset)):
        img = dataset[index][0]
        M = T@homographies[index]
        warped_list.append(cv2.warpPerspective(img, M, (wc,hc)))
    warped_list.append(cv2.warpPerspective(ref_img, T, (wc,hc)))

    for i in range(len(warped_list)):
        warped_list[i] = warped_list[i]//len(warped_list)
    
    finalImage = sum(warped_list)
    cv2.imwrite('output.jpg', finalImage)
    print("stitching complete!")
        

if __name__ == '__main__':
    main()
    
