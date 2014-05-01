import random
import math
import cPickle as pickle
from RBNeuron import RBNeuron

class Trainer(object):
    
    n_versions = [2,3,4,5,6]
    training_iterations = 1000
    learningRate = 0.01
    
    
    def __init__(self, nn):

        self.nn = nn
        self.validationError = 0
        self.oldValidationError = 0
        


    def trainMLP(self, numInputs):
        totalError = 0
        i = 1
        epoch = 0
        
        with open("data/trainingSet" + numInputs, 'r') as f:
            
            while True:
                for line in f:
                    data = line.strip()
                    data = data.split(' ')
                    inputs = data[:-1]
                    expectedOutput = float(data[-1])
                    
                    self.nn.feedForward(inputs)
                    self.nn.backpropagate(expectedOutput)
                    self.nn.updateWeights(self.learningRate, enableMomentum=1)
                    
                    outputNeuron = self.nn.outputNeuron
                    percentError = (abs(outputNeuron.outputValue - expectedOutput)/expectedOutput)
                    
                    totalError += percentError
                    avgError = totalError/i
                    print 'Training... epoch: {0:<5d}  |   Percent Err:{1:>10.2%}   |   Avg Percent Err:{2:>10.2%}   |   Validation Err:{3:>10.2%}'.format(epoch, percentError, avgError, self.validationError)
                   
                    i += 1
                    
                # Return fp to head of file 
                f.seek(0)
                
                
                if epoch % 20 == 0:
                    self.oldValidationError = self.validationError
                    self.validate(numInputs)
                    
                        
                epoch += 1
                if self.validationError > self.oldValidationError and self.oldValidationError != 0: break    
            f.closed
            
                    
                    
        
    def trainRB(self, numInputs):
    
        # Create array of input space
        input_space = []
        with open("data/trainingSet" + numInputs, 'r') as f:
            for line in f:
                data = line.strip()
                data = data.split(' ')
                inputs = data[:-1]
                for i in inputs:
                    input_space.append(i)
                
            f.closed
        
        
        # Get centers from input space
        centers = random.sample(set(input_space), self.nn.numRadialBasisNeurons * self.nn.numInputNeurons)
        
        
        # Calculate maximum distance between points
        dmax = 0
        for i in centers:
            for j in centers:
                if abs(float(i)-float(j)) > dmax:
                    dmax = abs(float(i)-float(j))
        
        
        # Calculate width
        RBNeuron.width = dmax/math.sqrt(self.nn.numRadialBasisNeurons)
        
        
        
        # Make weights of input -> hidden edges random values from input space as centers of rb
        for rb in self.nn.radialBasisLayer:
            for edge in rb.inputs:
                edge.weight = float(random.sample(set(input_space), 1)[0])
            
        
        
        # Loop through training file and train RB net
        totalError = 0
        i = 1
        epoch = 0
        with open("data/trainingSet" + numInputs, 'r') as f:
            while True:
                for line in f:
                    data = line.strip()
                    data = data.split(' ')
                    inputs = data[:-1]
                    expectedOutput = float(data[-1])
                    
                    self.nn.feedForward(inputs)
                    self.nn.backpropagate(expectedOutput)
                    self.nn.updateWeights(self.learningRate, enableMomentum=1)
                    self.nn.updateParams(self.learningRate, enableMomentum=1)
                    
                    outputNeuron = self.nn.outputNeuron
                    percentError = (abs(outputNeuron.outputValue - expectedOutput)/expectedOutput)
                    
                    totalError += percentError
                    avgError = totalError/i
                    print 'Training... epoch: {0:<5d}  |   Percent Err:{1:>10.2%}   |   Avg Percent Err:{2:>10.2%}   |   Validation Err:{3:>10.2%}'.format(epoch, percentError, avgError, self.validationError)
                   
                    i += 1
                    
                # Return fp to head of file 
                f.seek(0)
                
                if epoch % 20 == 0:
                    self.oldValidationError = self.validationError
                    self.validate(numInputs)
                    
                epoch += 1
                if self.validationError > self.oldValidationError and self.oldValidationError != 0: break
                
            f.closed
       
 
 

    def validate(self, numInputs):
        i = 1
        totalValidationError = 0
        with open("data/validationSet" + numInputs, 'r') as f:
            for line in f:
                data = line.strip()
                data = data.split(' ')
                inputs = data[:-1]
                expectedOutput = float(data[-1])
                
                self.nn.feedForward(inputs)
                
                outputNeuron = self.nn.outputNeuron
                percentError = abs(outputNeuron.outputValue - expectedOutput)/expectedOutput
                
                totalValidationError += percentError
                avgError = totalValidationError/i
                
                # Keep track of validation error for early stopping technique to prevent overfitting
                self.validationError = avgError
                print 'Validating... Percent Error:{0:>10.2%}  |  Avg Percent Error:{1:>10.2%}'.format(percentError, avgError)
                i += 1
                            
            
            f.closed
    
    
    
    def generateTrainingSetFile(self):
        
        # Create training set file for each version
        for n in self.n_versions:
            # Open file for writing
            with open('data/trainingSet'+str(n), 'w') as f:
            
                for _ in range(0, self.training_iterations):
                    # Randomize input vector
                    x = [random.uniform(0, 0.25) for _ in range(n)]
                    
                    # Rosenbrock Function
                    expected = sum(((1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2) for i, _ in enumerate(range(0, n-1)))
    #                 expected = x[0] + x[1]
                      
                    for i, _ in enumerate(range(n)): 
                        f.write(str(x[i]) + " ")
                        
                    f.write(str(expected) + " ")
                    f.write("\n")
                
                f.close()
                
            # Open file for writing
            with open('data/validationSet'+str(n), 'w') as f:
            
                for _ in range(0, self.training_iterations/3):
                    # Randomize input vector
                    x = [random.uniform(0, 0.25) for _ in range(n)]
                    
                    # Rosenbrock Function
                    expected = sum(((1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2) for i, _ in enumerate(range(0, n-1)))
    #                 expected = x[0] + x[1]
                      
                    for i, _ in enumerate(range(n)): 
                        f.write(str(x[i]) + " ")
                        
                    f.write(str(expected) + " ")
                    f.write("\n")
                
                f.close()
                
            # Open file for writing
            with open('data/testSet'+str(n), 'w') as f:
            
                for _ in range(0, self.training_iterations/3):
                    # Randomize input vector
                    x = [random.uniform(0, 0.25) for _ in range(n)]
                    
                    # Rosenbrock Function
                    expected = sum(((1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2) for i, _ in enumerate(range(0, n-1)))
    #                 expected = x[0] + x[1]
                      
                    for i, _ in enumerate(range(n)): 
                        f.write(str(x[i]) + " ")
                        
                    f.write(str(expected) + " ")
                    f.write("\n")
                
                f.close()
