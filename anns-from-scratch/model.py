import numpy as np


class Model:
    def __init__(self, architecture, loss):
        self.architecture=architecture

        self.loss = loss

    def predict(self, x):
        output = x

        for layer in self.architecture:
            # new output is calculated by forwarding previous output through the layer
            output=layer.forward(output)
            
        return output
    

    def fit(self, X, Y, epochs=100, learning_rate=0.01, verbose=True):
        for e in range(epochs):

            error=0

            for x, y in zip(X, Y):
                # calculating output
                output = self.predict(x)

                # calculating loss
                error += self.loss.calculate_loss(output, y)
                gradient = self.loss.calculate_loss_derivative(output, y)

                # backpropagation
                self._backpropagation(gradient, learning_rate)
            
            # calculating loss
            error = error/len(X)

            # if verbose print loss in each epoch
            if verbose:
                print(f"Epoch {e+1}: Loss={error}")
        

    def _backpropagation(self, gradient, learning_rate):
        for layer in reversed(self.architecture):
            gradient = layer.backward(gradient, learning_rate)
