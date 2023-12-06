# Neural-Network-from-scratch

This is a simple neural network library implemented in Python only using NumPy. It is designed for educational purposes and easy integration into various projects. The library includes basic components such as neural network layers, activation functions, loss functions, and optimization algorithms.


## Prerequisites

Make sure you have the following prerequisites installed:
- Python (version 3.11.3)
- NumPy

## Files

### 1.Net
#### NeuralNet Class:
Purpose: Represents a complete feedforward neural network.<br />
Description: The NeuralNet class orchestrates the overall functioning of the neural network. It handles the forward and backward passes, updates parameters using gradient descent, and provides a method for making predictions. The network is composed of layers, each responsible for specific computations.

### 2.Layers
#### Linear_Layer Class:
Purpose: Represents a linear layer without bias.<br />
Description: The Linear_Layer class defines a basic linear layer in a neural network, responsible for computing the weighted sum of inputs. It includes methods for computing gradients during backpropagation, derivatives, and the layer's output.

#### Linear_Layer_Bias Class:
Purpose: Represents a linear layer with bias.<br />
Description: Building on the Linear_Layer class, Linear_Layer_Bias introduces bias terms to the computation. It includes additional methods to compute gradients for bias terms during backpropagation and produces the layer's output.

### 3.LossFunctions
#### MSE_Loss Class:
Purpose: Represents the Mean Squared Error (MSE) loss function.<br />
Description: The MSE_Loss class encapsulates the computation of the Mean Squared Error loss, a common objective function for regression tasks. It computes the loss and its derivative, facilitating the training process.

#### Softmax_CEL Class:
Purpose: Represents the Softmax Cross-Entropy loss function.<br />
Description: The Softmax_CEL class focuses on classification tasks, utilizing the Softmax activation and Cross-Entropy loss. It computes the loss and its derivative, providing a measure of how well the predicted probabilities match the true class labels.

### 4.Optimizers
#### Momentum Class:
Purpose: Represents stochastic gradient descent with momentum.<br />
Description: The Momentum class implements the stochastic gradient descent optimization algorithm with momentum. It maintains momentum terms for each parameter, helping accelerate convergence during training. The class integrates with a neural network to optimize its parameters.

### 5.Activations
#### ReLU Class:
Purpose: Represents the Rectified Linear Unit (ReLU) activation function.<br />
Description: The ReLU class in Activations.py defines the ReLU activation function, commonly used in neural networks to introduce non-linearity. It applies an element-wise rectification, setting negative values to zero and leaving positive values unchanged.

### 6.Helper
#### Softmax Activation Function:
Purpose: The softmax function implements the softmax activation, which is commonly used in neural networks for multi-class classification problems. It transforms the raw input scores into probabilities, making it suitable for interpreting the output as class probabilities.
