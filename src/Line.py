'''
Created on 06.02.2017

@author: christian
'''
import numpy as np
from Buffer import Buffer
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.x_length = 1280
        self.y_length = 720
        self.plot = np.linspace(0, self.y_length-1, self.y_length)
        self.n = 10
        # was the line detected in the last iteration?
        self.detected = False 
        # enough pixels
        self.enough_pixels = False  
        #average bottom x values of the fitted line over the last n iterations
        self.avg_x = Buffer((self.n, 1), self.n)     
        #polynomial coefficients averaged over the last n iterations
        self.avg_fit = Buffer((self.n, 3), self.n)  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.zeros((3), dtype='float') 
        #polynomial coefficients for the last recent fit
        self.last_fit = np.zeros((3), dtype='float') 
        #radius of curvature of the line in meters
        self.radius_of_curvature = 0 
        #distance in meters of vehicle center from the line
        self.line_from_center = 0 
        #difference in fit coefficients between last and new fits
        self.diffs = np.zeros((3), dtype='float')         
        #x values for current detected line pixels
        self.allx = None  
        #y values for current detected line pixels
        self.ally = None
        #count boxes with part of line
        self.boxCount = np.zeros((20))  
######################################################################################################
    def reset(self):
        self.boxCount = np.zeros((20))  
######################################################################################################       
    def calculateCurvation(self):
        y_eval = np.max(self.plot)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.plot*self.ym_per_pix, self.ptsFitLine(self.plot)*self.xm_per_pix, 2)
        # Calculate the new radius of curvature
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) \
                                     / np.absolute(2*fit_cr[0])
                                     
######################################################################################################
    def fitLine(self):
        if self.enough_pixels :
            self.current_fit = np.polyfit(self.ally , self.allx, 2)
######################################################################################################
    def evaluateLine(self, otherLine):
        if self.setAndEvaluateDiffs(self.current_fit, self.last_fit) and self.checkParallel(otherLine):
            self.addFit(self.current_fit)
            self.calculateLinePosition()
            self.detected = True
            self.last_fit = self.current_fit
        else:
            self.detected = False
######################################################################################################        
    def setAndEvaluateDiffs(self, currentFit, lastFit):      
        if not lastFit.any():
            return True
                       
        self.diffs = np.absolute(currentFit - lastFit)
        
        if (self.diffs[0] > 1.0e-3) or (self.diffs[1] > 1.0) or (self.diffs[2] > 2.5e2):
            print("\nDiffs:", self.diffs)
            return False
        return True
######################################################################################################        
    def addFit(self, fit):
        self.avg_fit.insert(fit)
        self.avg_x.insert(self.getBasePositionX(fit))
######################################################################################################        
    def getBasePositionX(self, fit):
        pos = fit[0]*(self.y_length-1)**2 + fit[1]*(self.y_length-1) + fit[2]
        return pos  

######################################################################################################        
    def calculateLinePosition(self):
        self.line_from_center = ((self.x_length/2) - self.avg_x.avg()[0])*self.xm_per_pix
######################################################################################################          
    def setPixels(self, x_pixels, y_pixels):
        minpix = 150
        if (len(x_pixels) < minpix):
            self.enough_pixels = False  
            return
        self.allx = x_pixels
        self.ally = y_pixels
        self.enough_pixels = True 
######################################################################################################    
    def ptsFitLine(self, delta=0):
        fit = self.avg_fit.avg()
        return fit[0]*self.plot**2 + fit[1]*self.plot + fit[2] + delta
######################################################################################################
    def checkParallel(self, line):
        X = np.array([0, 180, 360, 540, 720])
        dist = np.abs(self.getY(X) - line.getY(X))
        max_diff = np.amax(np.abs(dist - np.min(dist)))
        if max_diff > 250:
            return False
        return True 
             
######################################################################################################
    def mergeLines(self, line):
        pass
######################################################################################################
    def getXStart(self):
        return self.avg_x.avg()
######################################################################################################
    def getY(self, X ):
        return self.current_fit[0]*X**2 + self.current_fit[1]*X + self.current_fit[2]   