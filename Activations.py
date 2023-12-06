import numpy as np

class ReLU():
    def __init__(self):
        self.input = None

    def output(self, input):
        """
        Computes the output of the ReLU activation function.

        Parameters:
            input (numpy.matrix): Input data.

        Returns:
            numpy.matrix: Output after applying the ReLU activation.
        """
        self.input = input
        input[input<0] = 0
        return input
    
    def der(self, derivative):
        """
        Computes the derivative of the ReLU activation function.

        Parameters:
            derivative (numpy.matrix): Derivative of the loss with respect to the output.

        Returns:
            numpy.matrix: Derivative of the loss with respect to the input.
        """
        
        input = self.input
        input[input<0]=0
        input[input>0]=1
        return np.multiply(derivative, input.transpose())
    
    def grad(self, derivative):
        """
        Placeholder for gradient computation; not implemented for ReLU.
        Will be called in the backward method of the NeuralNet() class inside Net.py.

        Parameters:
            derivative (numpy.matrix): Derivative of the loss with respect to the output.
        """
        pass
