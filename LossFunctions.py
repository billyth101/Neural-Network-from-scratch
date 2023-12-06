import numpy as np
import Helper as hp

class MSE_Loss():
    """
    MSE_Loss class represents the Mean Squared Error (MSE) loss function.

    Attributes:
        input (np.matrix): Input data.
    """
    def __init__(self):
        self.input = None

    def output(self, input , label):
        """
        Computes the output of the MSE loss function.

        Parameters:
            input (np.matrix): Predicted output.
            label (np.matrix): Ground truth labels.

        Returns:
            float: Mean Squared Error loss.
        """
        self.input = input
        loss = 0
        for k in range(0, input.shape[0]):
            for n in range(0, input.shape[1]):
                loss = loss + (input[k,n]-label[k,n])**2
        return loss/(input.shape[1]*input.shape[0])
    
    def der(self, label):
        """
        Computes the derivative of the MSE loss function.

        Parameters:
            label (np.matrix): Ground truth labels.

        Returns:
            np.matrix: Derivative of the loss.
        """
        derivative = 2*(self.input - label).transpose()/(self.input.shape[1]*self.input.shape[0])
        return derivative
    
class Softmax_CEL():
    """
    Softmax_CEL class represents the Softmax Cross-Entropy Loss function.

    Attributes:
        input (np.matrix): Input data.
    """
    def __init__(self):
        self.input = None

    def output(self, input, label):
        """
        Computes the output of the Softmax Cross-Entropy loss function.

        Parameters:
            input (np.matrix): Predicted output.
            label (np.matrix): Ground truth labels.

        Returns:
            float: Softmax Cross-Entropy loss.
        """
        self.input = input
        val = hp.softmax(input)
        val = np.log(val+0.000001)
        val = np.multiply(label, val)
        val = -np.sum(val, axis=0)
        return np.sum(val)/input.shape[1]
    
    def der(self, label):
        """
        Computes the derivative of the Softmax Cross-Entropy loss function.

        Parameters:
            label (np.matrix): Ground truth labels.

        Returns:
            np.matrix: Derivative of the loss.
        """
        der = label.T
        mat = hp.softmax(self.input).T

        return mat - der