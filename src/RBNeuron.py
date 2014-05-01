import math
from Neuron import Neuron

class RBNeuron(Neuron):
    
    width = 0
        
    def __init__(self, value = float(0)):
        
        self.outputValue = value
        self.inputs = []
        self.outputs = []
        self.weightError = 0
        self.centerError = 0
        self.widthError = 0
        self.centerPoint = None
        
        
    
    def eval(self, width):
        if self.outputs == []:
            self.outputValue = sum(edge.source.outputValue * edge.weight for edge in self.inputs)
        else:
            self.outputValue = self.radialBasisFunction(self.calculateNorm(), width)
            
    
    def calculateNorm(self):
        norm = sum([(edge.source.outputValue - edge.weight)**2 for edge in self.inputs])
        return norm
    
    
    
    def updateWeights(self, learningRate, enableMomentum):
        for edge in self.inputs:
            gradient = edge.target.weightError * edge.source.outputValue
            edge.weight += (learningRate * gradient) + ((edge.previousWeightChange * edge.momentumConstant) if enableMomentum else 0)
            edge.previousWeightChange = learningRate * gradient
    
    
    def updateCenter(self, learningRate, enableMomentum):
        for edge in self.inputs:
            gradient = edge.target.weightError * edge.source.outputValue
            edge.weight += (learningRate * gradient) + ((edge.previousCenterChange * edge.momentumConstant) if enableMomentum else 0)
            edge.previousCenterChange = learningRate * gradient
    
 
    
    def calculateWeightError(self, expectedOutput = None):
        
        self.weightError = expectedOutput - self.outputValue
        return self.weightError
    
    
    
    def calculateCenterError(self, expectedOutput = None):
        
        self.centerError = expectedOutput - self.outputValue
        
        return self.centerError
    
    
    
    def calculateWidthError(self, expectedOutput = None):
        
        self.widthError = expectedOutput - self.outputValue
        
        return self.widthError
    
    
    
    def radialBasisFunction(self, eNorm, width):
        try:
            return float(math.e**(-eNorm/(width**2)))
 
        except OverflowError:
            return float("inf")

