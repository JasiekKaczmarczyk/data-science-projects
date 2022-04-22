import numpy as np

class Model:
    def __init__(self, architecture, loss):
        self.architecture=architecture

        self.loss = loss

    def predict(self, x):
        # transposing x [batch_size, features] -> [features, batch_size]
        output = x.T

        for layer in self.architecture:
            # new output is calculated by forwarding previous output through the layer
            output=layer.forward(output)
        
        # transposing output back [features, batch_size] -> [batch_size, features]
        return output.T
    

    def fit(self, X, Y, batch_size, epochs=100, learning_rate=0.01, verbose=True):
        for e in range(epochs):

            error=0

            X_shuffled, Y_shuffled = self._shuffle_data(X, Y)
            X_shuffled, Y_shuffled = self._create_batches(X_shuffled, Y_shuffled, batch_size)

            for x, y in zip(X_shuffled, Y_shuffled):
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

    def _shuffle_data(self, X, Y):
        shuffled_indices = np.random.permutation(len(X))

        return X[shuffled_indices], Y[shuffled_indices]
    
    def _create_batches(self, X, Y, batch_size):
        return np.split(X, range(batch_size, len(X), batch_size)), np.split(Y, range(batch_size, len(Y), batch_size))
