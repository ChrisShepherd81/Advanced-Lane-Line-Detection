'''
Created on 06.02.2017

@author: christian
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import datetime
from Line import Line
import glob

def findLines(warpedImage, leftLine, rightLine, showImages = False):
    #print(warpedImage.shape)
    binary_warped = np.zeros((warpedImage.shape[0],warpedImage.shape[1] ))
    binary_warped[(warpedImage > 0)] = 1

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype(np.uint8)
    
    if not leftLine.detected or not rightLine.detected: 
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/1.5):,:], axis=0) #save surens of line
        if showImages:
            plt.plot(histogram)
            plt.show()
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        leftx_base = leftLine.getXStart()
        rightx_base = rightLine.getXStart()
    # Choose the number of sliding windows
    nwindows = 20
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each wdw_ind
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter wdw_ind
    minpix = 20
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    leftLine.reset()
    rightLine.reset()
    # Step through the windows one by one
    for wnd_ind in range(nwindows):
        # Identify wdw_ind boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (wnd_ind+1)*window_height
        win_y_high = binary_warped.shape[0] - wnd_ind*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
            
        # Identify the nonzero pixels in x and y within the wdw_ind
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
           
        # Draw the windows on the visualization image
        if len(good_left_inds) > 0:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 4) 
            leftLine.boxCount[wnd_ind] = 1
        else:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,0), 2)
            rightLine.boxCount[wnd_ind] = 1
            
        # Draw the windows on the visualization image
        if len(good_right_inds) > 0:
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 4) 
        else:
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,0,0), 2) 
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next wdw_ind on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftLine.setPixels(nonzerox[left_lane_inds], nonzeroy[left_lane_inds] )
    rightLine.setPixels(nonzerox[right_lane_inds], nonzeroy[right_lane_inds])
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Fit a second order polynomial to each
    leftLine.fitLine()
    rightLine.fitLine()
    
    leftLine.evaluateLine(rightLine)
    rightLine.evaluateLine(leftLine)
          
    left_fitx = leftLine.ptsFitLine()
    right_fitx = rightLine.ptsFitLine()
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    if showImages:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpedImage).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    color = (0,255, 0)
#     imageName = "../failed/fail_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
#     if not leftLine.detected:
#         color = (255,0, 0)
#     if not rightLine.detected:
#         color = (255,0, 0)
#             
#     if color == (255,0,0):
#         print(imageName)
#         mpimg.imsave(imageName, np.dstack((warpedImage, warpedImage, warpedImage)))
        
    cv2.fillPoly(color_warp, np.int_([pts]), color)
    
    return color_warp

if __name__ == '__main__':

    #images = glob.glob('../failed/fail*.jpg')
    images = glob.glob('../test_images/test*.jpg')
    
    for idx, fname in enumerate(images):
        left = Line()
        right = Line()
        image = mpimg.imread(fname)
        binary = np.zeros((image.shape[0], image.shape[1]))
        binary[image[:,:,0] > 0] = 1        
        findLines(binary, left, right, True)