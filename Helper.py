import numpy as np

def softmax(input):
    """
    Applies the softmax activation function to the input array.

    Parameters:
        input (numpy.ndarray): Input array or matrix.

    Returns:
        numpy.ndarray: Output array after applying the softmax activation.
    """
    val = np.exp(input)
    return val/np.sum(val, axis=0)