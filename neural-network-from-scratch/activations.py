import numpy as np
from layers import Layer


class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation=activation
        self.activation_derivative=activation_derivative

    def forward(self, x):
        self.x=x

        # returning activation of the input
        return self.activation(x)

    def backward(self, output_gradient, learning_rate):
        # returning output gradient for previous layer
        return output_gradient*self.activation_derivative(self.x)

class Tanh(Activation):
    """
    Initailizes Tanh activation layer
    """
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_derivative = lambda x: 1-np.tanh(x)**2

        super().__init__(activation=tanh, activation_derivative=tanh_derivative)

class Sigmoid(Activation):
    """
    Initailizes Sigmoid activation layer
    """
    def __init__(self):
        sigmoid = lambda x: 1/(1+np.exp(-x))
        sigmoid_derivative = lambda x: sigmoid(x)*(1-sigmoid(x))

        super().__init__(activation=sigmoid, activation_derivative=sigmoid_derivative)

class ReLU(Activation):
    """
    Initailizes ReLU activation layer
    """
    def __init__(self):
        relu = lambda x: x * (x>0)
        relu_derivative = lambda x: 1.0*(x>0)

        super().__init__(activation=relu, activation_derivative=relu_derivative)

class LeakyReLU(Activation):
    """
    Initailizes LeakyReLU activation layer

    Parameters:
        alpha: float
            leakage parameter
    """
    def __init__(self, alpha):
        relu = lambda x: np.where(x>=0, x, alpha*x)
        relu_derivative = lambda x: np.where(x>=0, 1, alpha)

        super().__init__(activation=relu, activation_derivative=relu_derivative)