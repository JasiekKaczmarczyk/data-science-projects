import numpy as np

class Layer:
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    """
    Initializes Dense layer

    Parameters:
        input_size: intgi
            length of input features
        output_size: int
            length of output features
    """

    def __init__(self, input_size, output_size):
        # setting up weights and bias
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((1, output_size))
    
    def forward(self, x):
        """
        Forward pass of Dense

        Parameters:
            x: array
                array of shape [input_size, batch_size]

        Returns:
            output: array
                array of shape [output_size, batch_size]
        """

        # assigning input to self.x
        self.x=x

        # calculating forward step
        return np.dot(x, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass of Dense layer

        Parameters:
            output_gradient: array
                gradient outputed from next layer
            learning_rate: float
                determines how quickly network learns
        Returns:
            previous_x_gradient: array
                gradient which will be outputted to previous layer
        """

        # calculating gradient for weights
        weights_gradient=np.dot(self.x.T, output_gradient)
        bias_gradient=np.sum(output_gradient, axis=0, keepdims=True)

        # calculating gradient for previous layer (input layer)
        output_gradient=np.dot(output_gradient, self.weights.T)

        # gradient descent
        self.weights-=learning_rate*weights_gradient
        self.bias-=learning_rate*bias_gradient
        
        # returning output gradient for previous layer
        return output_gradient


class Dropout(Layer):
    """
    Initializes Dropout layer

    Parameters:
        dropout_rate: float
            probability of applying dropout for each input value
    """
    
    def __init__(self, dropout_rate=0.1):
        self.x = None
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        Forward pass of Dropout

        Parameters:
            x: array

        Returns:
            output: array
                array after dropout
        """
        self.x = x

        # creating mask of values to drop
        mask = np.random.uniform(size=x.shape) < self.dropout_rate

        # applying dropout
        self.x[mask] = 0.0

        # scaling x by (1-dropout_rate) to compensate for zeroed out values
        self.x = self.x*(1.0 - self.dropout_rate)

        return self.x

    def backward(self, output_gradient, learning_rate):
        return output_gradient