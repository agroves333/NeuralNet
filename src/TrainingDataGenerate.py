'''
Created on Nov 14, 2013

@author: Adam
'''
import unittest
from Trainer import Trainer
from MLPNN import MLPNN

class Test(unittest.TestCase):


    def testName(self):
        
        trainer = Trainer(MLPNN(1, 1))
        trainer.generateTrainingSetFile()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()