'''
Created on Sep 27, 2013

@author: Adam
'''

import random
import math
from RBNeuron import RBNeuron

class Tester(object):
    
    n_versions = [2,3,4,5,6]
    
    
    def __init__(self, nn):
        self.nn = nn    
        self.totalError = 0
      
    def test(self, numInputs):
        
        i = 1
        with open("data/testSet" + numInputs, 'r') as f:
            for line in f:
                data = line.strip()
                data = data.split(' ')
                inputs = data[:-1]
                expectedOutput = float(data[-1])
                
                self.nn.feedForward(inputs)
                
                outputNeuron = self.nn.outputNeuron
                percentError = abs(outputNeuron.outputValue - expectedOutput)/expectedOutput
                
                self.totalError += percentError
                avgError = self.totalError/i
                print 'Testing... Percent Error:{0:>10.2%}  |  Avg Percent Error:{1:>10.2%}'.format(percentError, avgError)
                
                i += 1
                              
            
            f.closed
            
            
    def testRB(self, numInputs):
        
        i = 1
        with open("data/testSet" + numInputs, 'r') as f:
            for line in f:
                data = line.strip()
                data = data.split(' ')
                inputs = data[:-1]
                expectedOutput = float(data[-1])
                
                self.nn.feedForward(inputs, RBNeuron.width)
                
                outputNeuron = self.nn.outputNeuron
                percentError = abs(outputNeuron.outputValue - expectedOutput)/expectedOutput
                
                self.totalError += percentError
                avgError = self.totalError/i
                print 'Testing... Percent Error:{0:>10.2%}  |  Avg Percent Error:{1:>10.2%}'.format(percentError, avgError)
                
                i += 1
                              
            
            f.closed
