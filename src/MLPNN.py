'''
Created on Sep 2, 2013

@author: adam
'''
from NeuralNetwork import NeuralNetwork
from Neuron import Neuron
from Edge import Edge

class MLPNN(NeuralNetwork):
    
    lastLearningRate = 0
    
    def __init__(self, numInputNeurons, numHiddenLayers):

        # Create Layers
        self.inputLayer = [Neuron() for _ in range(numInputNeurons)]
        self.numHiddenLayers = numHiddenLayers
        self.hiddenLayers = [[Neuron() for _ in range(numInputNeurons)] for _ in range(numHiddenLayers)]
        self.outputNeuron = Neuron()
        
        if numHiddenLayers > 0:
            # Create edges to connect input neurons to first hidden layer neurons (if exists)
            for i_neuron in self.inputLayer:
                for h_neuron in self.hiddenLayers[0]:
                    Edge(i_neuron, h_neuron)
            
          
             
            # Create edges to connect hidden layer neurons to each other
            for h_layer1, h_layer2 in [(self.hiddenLayers[i], self.hiddenLayers[i+1]) for i in range(self.numHiddenLayers-1)]:
                for h_neuron1 in h_layer1:
                    for h_neuron2 in h_layer2:
                        Edge(h_neuron1, h_neuron2)
                           
                         
            # Create edges to connect last hidden layer neurons to output neuron
            for h_neuron in self.hiddenLayers[-1]:
                Edge(h_neuron, self.outputNeuron)
        
       
        else:
            # Create edges to connect input neurons to output neuron
            for i_neuron in self.inputLayer:
                Edge(i_neuron, self.outputNeuron)
    
    
    
       
    def feedForward(self, inputs):
        # Feed input vector to input neurons
        for i, x in enumerate(inputs):
            self.inputLayer[i].outputValue = float(x)
        
        if self.numHiddenLayers > 0:
            # Evaluate hidden layers based on input layer outputs
            for layer in self.hiddenLayers:
                for neuron in layer:
                    neuron.eval()
                    
            
        # Evaluate output neuron based on last hidden layer's outputs
        self.outputNeuron.eval()
        
        
    
    
    
    def backpropagate(self, expectedOutput):
        self.outputNeuron.calculateWeightError(expectedOutput)
        
        if self.numHiddenLayers > 0:
            for neuron in self.hiddenLayers[-1]:
                neuron.calculateWeightError(self.outputNeuron.weightError)
                
            
            for layer in reversed(self.hiddenLayers):
                for neuron in layer:
                    neuron.calculateWeightError()
                    
              
        for neuron in self.inputLayer:
            neuron.calculateWeightError()
            

    
    def updateWeights(self, learningRate, enableMomentum):
        if self.numHiddenLayers > 0:
            for layer in self.hiddenLayers:
                for neuron in layer:
                    neuron.updateWeights(learningRate, enableMomentum)
                   
        self.outputNeuron.updateWeights(learningRate, enableMomentum)
        
            
    def printOutput(self):
        print "Error: " + str(self.outputNeuron.weightError) + "  Value: " + str(self.outputNeuron.outputValue) + "  ID: " + str(self.outputNeuron.id)
        

    


    
    
        