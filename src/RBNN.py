from NeuralNetwork import NeuralNetwork
from Neuron import Neuron
from Edge import Edge
from RBNeuron import RBNeuron
# from Tkinter import *


class RBNN(NeuralNetwork):
    
    lastLearningRate = 0
    
    def __init__(self, numInputNeurons, numRadialBasisNeurons):
   
        # Create Layers
        self.inputLayer = [Neuron() for _ in range(numInputNeurons)]
        self.radialBasisLayer = [RBNeuron() for _ in range(numRadialBasisNeurons)]
        self.outputNeuron = Neuron()
        self.numRadialBasisNeurons = numRadialBasisNeurons
        self.numInputNeurons = numInputNeurons
        
        # Create edges to connect input neurons to centerPoint layer neurons
        for i_neuron in self.inputLayer:
            for r_neuron in self.radialBasisLayer:
                Edge(i_neuron, r_neuron)
        
        # Create edges to connect centerPoint layer neurons to output neuron
        for r_neuron in self.radialBasisLayer:
            Edge(r_neuron, self.outputNeuron)
    
    

        
    def feedForward(self, inputs):
        # Feed input vector to input neurons
        for i, x in enumerate(inputs):
            self.inputLayer[i].outputValue = float(x)
        
        
        for neuron in self.radialBasisLayer:
            neuron.eval(neuron.width)
        
        # Evaluate output neuron based on last hidden layer's outputs
        self.outputNeuron.eval()
    
    
    
    def backpropagate(self, expectedOutput):
        self.outputNeuron.calculateWeightError(expectedOutput)
        
        for neuron in self.radialBasisLayer:
            neuron.calculateCenterError(expectedOutput)
            


  
    def updateWeights(self, learningRate, enableMomentum):
        self.outputNeuron.updateWeights(learningRate, enableMomentum)
        
        
        
    def updateParams(self, learningRate, enableMomentum):
        for neuron in self.radialBasisLayer:
            neuron.updateCenter(learningRate, enableMomentum)
        

    def printOutput(self):
        print "Error: " + str(self.outputNeuron.weightError) + "  Value: " + str(self.outputNeuron.outputValue) + "  ID: " + str(self.outputNeuron.id)
        

        
        
    def draw(self):
        root = Tk()
        canvas = Canvas(root, width=700, height=800)
        
        canvas.pack()
        
        '''
        Calculate offsets
        '''
        win_height = int(canvas.cget("height"))
        
        input_layer_y_offset = (win_height-len(self.inputLayer)*50)/(len(self.inputLayer)+1)        
        hidden_layer_y_offset = (win_height-len(self.radialBasisLayer)*50)/(len(self.radialBasisLayer)+1)
        output_neuron_y_offset = (win_height-50)/2
        
        '''
        Draw Neurons
        '''
        y_offset = input_layer_y_offset
        for neuron in self.inputLayer:
            position = 110, y_offset, 160, 50+y_offset
            canvas.create_oval(position, tag="neuron", fill="lightblue")
            canvas.create_text(position[0]+25, position[1]+25, text=str(neuron.outputValue), tag="outputValue")
            canvas.create_text(position[0]+30, position[1]-20, text="weightError: "+str(neuron.weightError))
            neuron.setNeuronCenter(*position)
            y_offset = y_offset + input_layer_y_offset + 50

        
        x_offset = 0
        y_offset = hidden_layer_y_offset
        
        for neuron in self.radialBasisLayer:
            position = 310+x_offset, y_offset, 360+x_offset, 50+y_offset
            canvas.create_oval(position, tag="neuron", fill="lightblue")
            canvas.create_text(position[0]+25, position[1]+25, text="{0:.20f}".format(neuron.outputValue), tag="outputValue")
            canvas.create_text(position[0]+30, position[1]-20, text="weightError: "+str(neuron.weightError))
            neuron.setNeuronCenter(*position)
            y_offset = y_offset + hidden_layer_y_offset + 50
        x_offset = x_offset + 200

        output_position = 310+x_offset, output_neuron_y_offset, 360+x_offset, 50+output_neuron_y_offset
        canvas.create_oval(output_position, tag="neuron", fill="lightblue")
        canvas.create_text(output_position[0]+25, output_position[1]+25, text=str(round(self.outputNeuron.outputValue, 2)), tag="outputValue")
        canvas.create_text(output_position[0]+30, output_position[1]-20, text="weightError: "+str(self.outputNeuron.weightError))
        self.outputNeuron.setNeuronCenter(*output_position)
            
        
         
        '''
        Draw Edges
        '''
        # Create edges to connect input neurons to radial basis layer neurons (if exists)
        for i_neuron in self.inputLayer:
            for h, h_neuron in enumerate(self.radialBasisLayer):
                canvas.create_line(i_neuron.centerPoint, h_neuron.centerPoint, tag="line")
                canvas.create_line(i_neuron.centerPoint, (i_neuron.centerPoint[0]-50, i_neuron.centerPoint[1]), tag="line")
                text_offset = (h_neuron.centerPoint[1]-i_neuron.centerPoint[1])*.25
                canvas.create_text(i_neuron.centerPoint[0]+100, i_neuron.centerPoint[1]+text_offset, text="w: "+str(i_neuron.outputs[h].weight))
                
        
        # Create edges to connect radial basis layer neurons to output neuron
        for h, h_neuron in enumerate(self.radialBasisLayer):
            canvas.create_line(h_neuron.centerPoint, self.outputNeuron.centerPoint, tag="line")
            text_offset = (self.outputNeuron.centerPoint[1]-h_neuron.centerPoint[1])*.25
            canvas.create_text(h_neuron.centerPoint[0]+100, h_neuron.centerPoint[1]+text_offset, text="w: "+str(h_neuron.outputs[0].weight))  
        
        
        canvas.lift("neuron", "line")
        canvas.lift("outputValue", "neuron")
         
        mainloop()  