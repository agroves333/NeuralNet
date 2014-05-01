import random

class Edge(object):
    
    def __init__(self, source, target):
        self.weight = float(round(random.uniform(-0.5, 0.5), 2))
        self.source = source
        self.target = target
        self.previousWeightChange = 0
        self.previousCenterChange = 0
        self.previousWidthChange = 0
        self.previousGradient = 0
        self.momentumConstant = 0.5
        
        source.getOutputs().append(self)
        target.getInputs().append(self)
    
    