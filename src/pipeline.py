'''
Created on 06.02.2017

@author: christian
'''
from imagePreProcessing import sobeling, binaryImage
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from lineFinder import findLines
import cv2
from ImageTransformer import ImageTransformer
from Line import Line

transformer = ImageTransformer((720,1280))
leftLine = Line()
rightLine = Line()

def pipeline(image, showPictures = False):
    global transformer
    global leftLine
    global rightLine
        
    if showPictures:
        plt.imshow(image)
        plt.show()
    
    undistImg = transformer.undistort(image)
    warpedImage = transformer.warp(undistImg)
    if showPictures:
        plt.imshow(warpedImage, cmap='gray')
        plt.show()
        
    edgesImage = binaryImage(warpedImage)
    if showPictures:
        plt.imshow(edgesImage, cmap='gray')
        plt.show()
    lineImage = findLines(edgesImage, leftLine, rightLine)
    if showPictures:
        plt.imshow(lineImage)
        plt.show()
    unwarpedImage = transformer.unwarp(lineImage)
    if showPictures:
        plt.imshow(unwarpedImage)
        plt.show()
        
    leftLine.calculateCurvation()
    rightLine.calculateCurvation()
    cv2.putText(unwarpedImage,"Curvature left: " + "{:.2f}".format(leftLine.radius_of_curvature) + "m right: " + "{:.2f}".format(rightLine.radius_of_curvature) + "m", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)
    cv2.putText(unwarpedImage,"Car pos: " + "{:.2f}".format(leftLine.line_from_center + rightLine.line_from_center) + "m", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)
    
    # Combine the result with the original image
    result = cv2.addWeighted(undistImg, 1, unwarpedImage, 0.3, 0)
    if showPictures:   
        plt.imshow(result)
        plt.show()
    return result
    
if __name__ == '__main__':

    line = [0]
    line[0]
    images = glob.glob('../test_images/test*.jpg')
    
    for idx, fname in enumerate(images):
        print(fname)
        image = mpimg.imread(fname)
        pipeline(image)