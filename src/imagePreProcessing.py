'''
Created on 04.02.2017

@author: Christian
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from numpy import uint8

######################################################################################################
def binaryImage(image):
    showPictures = False
    sum_img = 2*thresholding(image) + sobeling(image)
       
    if showPictures:
        plt.imshow(sum_img, cmap='gray')
        plt.axis('off')
        plt.show()
        
    combined = np.zeros_like(sum_img)
    combined[sum_img >= 3] = 1
    
    if showPictures:
        plt.imshow(combined, cmap='gray')
        plt.axis('off')
        plt.show()
        
    return combined
 
######################################################################################################
def thresholding(image): 
    showPictures = False
    
    #R channel threshold
    r_channel = image[:,:,0]
    r_channel_tresh = channel_threshold(r_channel, threshold=(200,255))
    if showPictures:
        plt.imshow(r_channel_tresh, cmap='gray')
        plt.axis('off')
        plt.show()
     
    #Convert to YUV colorspace            
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV).astype(np.float)
    y_channel = yuv[:,:,0]
    u_channel = yuv[:,:,1]

    #Convert to HLS colorspace 
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    sl_channel = np.maximum(s_channel, l_channel)
    
    #U channel threshold
    u_channel_tresh = channel_threshold(u_channel, threshold=(50,120))
    if showPictures:
        plt.imshow(u_channel_tresh, cmap='gray')
        plt.axis('off')
        plt.show()
        
    #Y channel threshold
    y_channel_tresh = channel_threshold(y_channel, threshold=(200,255))
    if showPictures:
        plt.imshow(y_channel_tresh, cmap='gray')
        plt.axis('off')
        plt.show()
        
    #max(S,L) channel threshold    
    sl_channel_tresh = channel_threshold(sl_channel, threshold=(200,255))    
    if showPictures:
        plt.imshow(sl_channel_tresh, cmap='gray')
        plt.axis('off')
        plt.show()
    
    #Sum all single binary images
    sum_img = u_channel_tresh + y_channel_tresh + sl_channel_tresh + r_channel_tresh
    
    return sum_img

######################################################################################################  
def sobeling(image):
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float)
  
    showPictures = False
   
    # Abs sobel X
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(15, 100))
    if showPictures:
        plt.imshow(gradx, cmap='gray')
        plt.axis('off')
        plt.show()
        
    # Abs sobel Y
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(15, 110))
    if showPictures:
        plt.imshow(grady, cmap='gray')
        plt.axis('off')
        plt.show()
    
    # Mag sobel    
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 200))
    if showPictures:
        plt.imshow(mag_binary, cmap='gray')
        plt.axis('off')
        plt.show()
    
    #Sum all single binary images
    sum_img = gradx + grady + mag_binary
    if showPictures:
        plt.imshow(sum_img, cmap='gray')
        plt.axis('off')
        plt.show()
        
    return sum_img
######################################################################################################  
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
       
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel) 
    # 3) Calculate the magnitude 
    sobelmag = np.sqrt(sobelx**2 + sobely**2)
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobelmag/np.max(sobelmag))
    
    # 5) Create a binary mask where mag thresholds are met
    sobelbinary = np.zeros_like(img)
    sobelbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sobelbinary
######################################################################################################
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
       
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros(img.shape)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return sbinary
######################################################################################################
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    sobel_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
    
    # 5) Create a binary mask where direction thresholds are met
    bin_sobel = np.zeros(image.shape)
    bin_sobel[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return bin_sobel

######################################################################################################
def channel_threshold(image, threshold=(150,255)):
    binary = np.zeros_like(image)
    binary[((image >= threshold[0]) & (image <= threshold[1]))] = 1
    return binary
######################################################################################################
