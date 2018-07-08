'''
Created on 11.02.2017

@author: Christian
'''
import numpy as np

class Buffer(object):
    '''
    classdocs
    '''
    def __init__(self, shape, size):
        '''
        Constructor
        '''
        self.size = size
        self.array = np.zeros(shape)
        self.count = 0
        self.index = 0
        
    def insert(self, vals):
        self.array[self.index] = vals
        self.update()
        
    def update(self):
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)
        
    def avg(self):
        return np.sum(self.array, axis=0) / self.count
        
        