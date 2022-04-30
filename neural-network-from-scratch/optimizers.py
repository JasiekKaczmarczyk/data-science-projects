import numpy as np
import layers

class Optimizer:
    def __init__(self):
        self.learning_rate = None

    def step(self, layer: layers.Dense):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
    
    def step(self, layer: layers.Dense):
        # updating weights and biases
        layer.weights-=self.learning_rate*layer.weights_gradient
        layer.bias-=self.learning_rate*layer.bias_gradient

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate: float, beta: float):
        self.learning_rate = learning_rate
        self.beta = beta
    
    def step(self, layer: layers.Dense):
        # calculating velocities
        layer.velocity_weights = self.beta*layer.velocity_weights + (1-self.beta)*layer.weights_gradient
        layer.velocity_bias = self.beta*layer.velocity_bias + (1-self.beta)*layer.bias_gradient

        # updating weights and biases
        layer.weights-=self.learning_rate*layer.velocity_weights
        layer.bias-=self.learning_rate*layer.velocity_bias

class RMSprop(Optimizer):
    def __init__(self, learning_rate: float, gamma: float):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.correction = 1e-8
    
    def step(self, layer: layers.Dense):
        # calculating velocities
        layer.velocity_weights = self.gamma*layer.velocity_weights + (1-self.gamma)*layer.weights_gradient**2
        layer.velocity_bias = self.gamma*layer.velocity_bias + (1-self.gamma)*layer.bias_gradient**2

        # updating weights and biases
        layer.weights-=self.learning_rate*layer.weights_gradient/(np.sqrt(layer.velocity_weights) + self.correction)
        layer.bias-=self.learning_rate*layer.bias_gradient/(np.sqrt(layer.velocity_bias) + self.correction)