import math

class Neuron(object):
        
    def __init__(self, value = float(0)):
        self.inputs = []
        self.outputs = []
        self.outputValue = value
        self.centerPoint = None
        self.weightError = 0
        
    
    def getOutputs(self):
        return self.outputs
    
    def getInputs(self):
        return self.inputs
    
    def getWeightError(self):
        return self.weightError
    
    def setWeightError(self, weightError):
        self.weightError = weightError
        
        
    def getOutputValue(self):
        return self.outputValue
    
    def eval(self):
        self.outputValue = self.activationFunction(sum(edge.source.outputValue * edge.weight for edge in self.inputs))
        

    def activationFunction(self, x):
        
        try:
            if self.outputs == []: # is output neuron
                return float(x)
            else:
                return float(1/(1+math.e**(-x)))
                
    
        except OverflowError:
            return float("inf")
        
        
    
    def updateWeights(self, learningRate, enableMomentum):
        for edge in self.inputs:
            gradient = edge.target.weightError * edge.source.outputValue
            edge.weight += (learningRate * gradient) + ((edge.previousWeightChange * edge.momentumConstant) if enableMomentum else 0)
            edge.previousWeightChange = learningRate * gradient

    
    
    def calculateWeightError(self, expectedOutput = None):
        
        if self.outputs == []:
            self.setWeightError(expectedOutput - self.outputValue)
        else:
            self.weightError = self.outputValue * (1.0 - self.outputValue) * sum([float(edge.weight) * float(edge.target.weightError) for edge in self.outputs])
        
        return self.weightError
    
    
    

    def setNeuronCenter(self, x1, y1, x2, y2):
        self.centerPoint = (x1+(x2-x1)/2, y1+(y2-y1)/2)
        