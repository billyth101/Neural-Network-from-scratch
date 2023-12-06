class NeuralNet():
    """
    NeuralNet class represents a simple feedforward neural network.

    Attributes:
        layers (list): List of Layer objects representing the layers in the neural network.

    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, input, label):
        """
        Performs forward pass through the neural network and computes the output.

        Parameters:
            input (numpy.matrix): Input data for the neural network.
            label (numpy.matrix): Ground truth labels for training.

        Returns:
            numpy.matrix: Output of the neural network.
        """
        out = input
        for i in range(len(self.layers)-1):
            out = self.layers[i].output(out)
        out = self.layers[-1].output(out, label)
        return out
    
    def backward(self, label):
        """
        Performs backward pass through the neural network and computes gradients.

        Parameters:
            label (numpy.matrix): Ground truth labels for training.
        """
        der = self.layers[-1].der(label)
        for i in range(len(self.layers)-2,-1,-1):
            self.layers[i].grad(der)
            der = self.layers[i].der(der)

    def step(self, rate):
        """
        Updates the parameters of the neural network using gradient descent.

        Parameters:
            rate (float): Learning rate for the gradient descent.
        """
        for i in range(len(self.layers)-1):
            self.layers[i].update(rate)

    def predict(self, input):
        """
        Performs a forward pass to make predictions on new data.

        Parameters:
            input (numpy.matrix): Input data for making predictions.

        Returns:
            numpy.matrix: Predicted output of the neural network.
        """
        out = input
        for i in range(len(self.layers)-1):
            out = self.layers[i].output(out)
        return out
    
    

      
