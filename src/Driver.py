from RBNN import RBNN
from MLPNN import MLPNN
from Trainer import Trainer
from Tester import Tester
from NeuralNetwork import NeuralNetwork


def main():

    print "Select Network Type";
    print "1. MLP"
    print "2. Radial Basis"
    nn_type = raw_input()
    nn = NeuralNetwork()
    
    if nn_type == "1":
        num_inputs = int(raw_input("Enter number inputs"))
        num_hidden = int(raw_input("Enter number hidden layers"))
        nn = MLPNN(num_inputs, num_hidden)
    
     
    elif nn_type == "2":
        num_inputs = int(raw_input("Enter number inputs"))
        num_centers = int(raw_input("Enter number radial basis functions"))
        nn = RBNN(num_inputs, num_centers);
     
    trainer = Trainer(nn)
    tester = Tester(nn)
       
       
    if nn_type == "1":
        trainer.trainMLP(str(num_inputs))
        tester.test(str(num_inputs))
    elif nn_type == "2":
        trainer.trainRB(str(num_inputs))
        tester.testRB(str(num_inputs))
       
#     nn.draw()
      
if __name__ == '__main__':
    main()