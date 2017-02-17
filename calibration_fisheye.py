#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibration.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    
    --debug:    ./output/
    --square_size: 1.0
    --pattern_size
    <image mask> '../yourdatafolder/img_name*.imgformat' 
                 example '../data/left*.jpg'

This version is adapted from samples of opencv calibrate.py
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

print(cv2.__version__)

# local modules
from common import splitfn

# built-in modules
import os

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    # args.setdefault('--size_1', 9)
    # args.setdefault('--size_2', 6)

    if not img_mask:
        raise ValueError('no image has been found')
    else:
        img_mask = img_mask[0]

    img_names = glob(img_mask)
    debug_dir = args.get('--debug')
    if not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))


    pattern_size = (7, 5)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    print(img_names)
    print('starting the process...')
    print ('number of images {}'.format(np.shape(img_names)))
    img_names2 = img_names[::15]
    for fn in img_names2:
        print('processing %s... ' % fn, end='')
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            
        if (debug_dir):
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            outfile = debug_dir + name + '_chess.tiff'
            cv2.imwrite(outfile, vis)

            if found:
                img_names_undistort.append(debug_dir + name)

        if not found:
            print('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print('ok')

    # calculate camera distortion
    print (np.shape(obj_points))
    print (np.shape(img_points))
    rvecs = []
    tvecs = []
    obj_points_array = np.float32(np.asarray(obj_points))
    img_points_array = np.float32(np.asarray(img_points))

    print (obj_points[0][0])
    print (obj_points_array.dtype)

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('')
    for fn in img_names:
        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.fisheye.undistortImage(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        path, name, ext = splitfn(fn)
        outfile = debug_dir + name + '_undistorted.jpg'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

   # cv2.destroyAllWindows()
