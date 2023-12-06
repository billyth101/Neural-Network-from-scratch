import numpy as np


class Linear_Layer():
    """
    Linear_Layer class represents a linear layer in a neural network without bias.

    Attributes:
        input (numpy.ndarray or np.matrix): Input data.
        gradient (list): List to store gradients during backpropagation.
        weights (list): List to store weights for the layer.
    """
    def __init__(self, dimensions, weights=None):
        """
        Parameters:
            dimensions (tuple): Tuple containing the dimensions of the layer.
            weights (list of np.matrix): List of weight matrices. If None, random weights are initialized.
        """
        self.input = None
        self.gradient = None
        if weights is None:
            self.weights = [np.matrix(np.random.normal(size=(dimensions[1], dimensions[0])))]
        else:
            self.weights = weights

    def grad(self, derivative):
        """
        Computes the gradient during backpropagation.

        Parameters:
            derivative (np.matrix): Derivative of the loss with respect to the output.
        """
        grad = [self.input.dot(derivative).transpose()/derivative.shape[0]]
        self.gradient = grad
    
    def der(self, derivative):
        """
        Computes the derivative of the layer.

        Parameters:
            derivative (np.matrix): Derivative of the loss with respect to the output.

        Returns:
            np.matrix: Derivative of the loss with respect to the input.
        """
        return derivative.dot(self.weights[0])
    
    def output(self, input):
        """
        Computes the output of the layer.

        Parameters:
            input (np.matrix): Input data.

        Returns:
            np.matrix: Output of the layer.
        """
        self.input = input
        return self.weights[0].dot(input)
    
      

class Linear_Layer_Bias():
    """
    Linear_Layer_Bias class represents a linear layer in a neural network with bias.

    Attributes:
        input (numpy.ndarray or np.matrix): Input data.
        gradient (list): List to store gradients during backpropagation.
        weights (list): List to store weights (including bias) for the layer.
    """
    def __init__(self, dimensions, weights=None):
        """
        Parameters:
            dimensions (tuple): Tuple containing the dimensions of the layer.
            weights (list of np.matrix): List of weight matrices (including bias). If None, random weights are initialized.
        """

        self.input = None
        self.gradient = [np.matrix(np.random.normal(scale=0.5, size=(dimensions[1], dimensions[0]))), np.matrix(np.random.normal(size=(dimensions[1], 1)))]
        if weights is None:
            self.weights = [np.matrix(np.random.normal(scale=0.5, size=(dimensions[1], dimensions[0]))), np.matrix(np.random.normal(size=(dimensions[1], 1)))]
        else:
            self.weights = weights

    def grad(self, derivative):
        """
        Computes the gradients during backpropagation.

        Parameters:
            derivative (np.matrix): Derivative of the loss with respect to the output.
        """
        grad1 = self.input.dot(derivative).transpose()/derivative.shape[0]
        self.gradient[0] = grad1

        grad2 = np.ones(self.input.shape[1]).dot(derivative).transpose()/derivative.shape[0]
        self.gradient[1] = grad2
    
    def der(self, derivative):
        """
        Computes the derivative of the layer.

        Parameters:
            derivative (np.matrix): Derivative of the loss with respect to the output.

        Returns:
            np.matrix: Derivative of the loss with respect to the input.
        """
        return derivative.dot(self.weights[0])
    
    def output(self, input):
        """
        Computes the output of the layer.

        Parameters:
            input (np.matrix): Input data.

        Returns:
            np.matrix: Output of the layer.
        """
        self.input = input
        return self.weights[0].dot(input) + self.weights[1]
    
