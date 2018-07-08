'''
Created on 04.02.2017

@author: Christian
'''
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
######################################################################################################
def ApplyChessBoardCorners(chessBoard_dim):
    objp = np.zeros((chessBoard_dim[0]*chessBoard_dim[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessBoard_dim[0], 0:chessBoard_dim[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        print(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessBoard_dim, None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints
######################################################################################################
def ApplyCalibration():
    objpoints, imgpoints = ApplyChessBoardCorners((9, 6))
    
    # Test undistortion on an image
    img = cv2.imread('../camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('../camera_cal/test_undist.jpg',dst)
    
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "../camera_cal/calibration.p", "wb" ) )

    return img, dst
    
######################################################################################################
def getCalibration():
    calibration_file = "../camera_cal/calibration.p"
    with open(calibration_file, mode='rb') as f:
        calibration = pickle.load(f)
    return calibration["mtx"], calibration["dist"]
    
######################################################################################################
if __name__ == '__main__':
    
    img, dst = ApplyCalibration()

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()
    