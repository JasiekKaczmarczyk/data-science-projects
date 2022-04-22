import numpy as np

class Layer:
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # setting up weights and bias
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, x):
        # assigning input to self.x
        self.x=x

        # calculating forward step
        return self.weights @ x + self.bias

    def backward(self, output_gradient, learning_rate):

        # calculating gradient for weights
        weights_gradient=(output_gradient @ self.x.T)
        bias_gradient=np.sum(output_gradient, axis=1, keepdims=True)

        # calculating gradient for previous layer (input layer)
        previous_x_gradient=self.weights.T @ output_gradient

        # gradient descent
        self.weights-=learning_rate*weights_gradient
        self.bias-=learning_rate*bias_gradient
        
        # returning output gradient for previous layer
        return previous_x_gradient