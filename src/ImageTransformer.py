'''
Created on 08.02.2017

@author: christian
'''
import cv2
from cameraCalibration import getCalibration
import numpy as np

class ImageTransformer(object):

    def __init__(self, shape):
        srcPoints = np.float32([[595,450],[685,450],[1122,720],[205,720]])
        dstPoints = np.float32([[300,0],  [shape[1]-300,0],[shape[1]-300,shape[0]], [300,shape[0]]])
        mtx, dist = getCalibration()
        self.Mtx  = mtx
        self.Dist = dist
        self.M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        self.MInv = cv2.getPerspectiveTransform( dstPoints, srcPoints)

######################################################################################################
    def undistort(self, image):
        undist = cv2.undistort(image, self.Mtx, self.Dist, None, self.Mtx)
        return undist
    
######################################################################################################
    def warp(self, image):
        srcPoints = np.float32([[595,450],[685,450],[1122,720],[205,720]])
        dstPoints = np.float32([[100,0],  [720-100,0],[720-100,1280], [100,1280]])
        #self.M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        warped = cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)  
        return warped
    
######################################################################################################
    def unwarp(self, image):
        unwarped = cv2.warpPerspective(image, self.MInv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)  
        return unwarped
  