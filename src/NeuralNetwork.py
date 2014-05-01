'''
Created on Sep 2, 2013

@author: adam
'''
from Neuron import Neuron
# from Tkinter import *

class NeuralNetwork(object):
    
    def __init__(self):
        pass
        
  
    def draw(self):
        root = Tk()
        canvas = Canvas(root, width=(self.numHiddenLayers+2)*250, height=800)
        
        canvas.pack()
        
        '''
        Calculate offsets
        '''
        win_height = int(canvas.cget("height"))
        
        input_layer_y_offset = (win_height-len(self.inputLayer)*50)/(len(self.inputLayer)+1)        
        hidden_layer_y_offset = (win_height-len(self.hiddenLayers[0])*50)/(len(self.hiddenLayers[0])+1)
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
        for i, h_layer in enumerate(range(self.numHiddenLayers)):
            y_offset = hidden_layer_y_offset
            
            for neuron in self.hiddenLayers[i]:
                position = 310+x_offset, y_offset, 360+x_offset, 50+y_offset
                canvas.create_oval(position, tag="neuron", fill="lightblue")
                canvas.create_text(position[0]+25, position[1]+25, text=str(round(neuron.outputValue,2)), tag="outputValue")
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
        # Create edges to connect input neurons to first hidden layer neurons (if exists)
        for i_neuron in self.inputLayer:
            for h, h_neuron in enumerate(self.hiddenLayers[0]):
                canvas.create_line(i_neuron.centerPoint, h_neuron.centerPoint, tag="line")
                canvas.create_line(i_neuron.centerPoint, (i_neuron.centerPoint[0]-50, i_neuron.centerPoint[1]), tag="line")
                text_offset = (h_neuron.centerPoint[1]-i_neuron.centerPoint[1])*.25
                canvas.create_text(i_neuron.centerPoint[0]+100, i_neuron.centerPoint[1]+text_offset, text="w: "+str(i_neuron.outputs[h].weight))
                
        # Create edges to connect hidden neurons next hidden layer neurons
        for h_layer1, h_layer2 in [(self.hiddenLayers[i], self.hiddenLayers[i+1]) for i in range(self.numHiddenLayers-1)]:
            for h1, h_neuron1 in enumerate(h_layer1):
                for h2, h_neuron2 in enumerate(h_layer2):
                    canvas.create_line(h_neuron1.centerPoint, h_neuron2.centerPoint, tag="line")
                    text_offset = (h_neuron2.centerPoint[1]-h_neuron1.centerPoint[1])*.25
                    canvas.create_text(h_neuron1.centerPoint[0]+100, h_neuron1.centerPoint[1]+text_offset, text="w: "+str(h_neuron1.outputs[h2].weight))  
                
        # Create edges to connect last hidden layer neurons to output neuron
        for h, h_neuron in enumerate(self.hiddenLayers[-1]):
            canvas.create_line(h_neuron.centerPoint, self.outputNeuron.centerPoint, tag="line")
            text_offset = (self.outputNeuron.centerPoint[1]-h_neuron.centerPoint[1])*.25
            canvas.create_text(h_neuron.centerPoint[0]+100, h_neuron.centerPoint[1]+text_offset, text="w: "+str(self.hiddenLayers[-1][h].outputs[h-1].weight))  
        
        
        canvas.lift("neuron", "line")
        canvas.lift("outputValue", "neuron")
         
        mainloop()
        
        
        
        
        
        