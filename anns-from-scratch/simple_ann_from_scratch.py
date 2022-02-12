import numpy as np

class ANN:   
    def __init__(self, layers=[3, 5, 3, 1]):
        """
        Constructor for Simple Artificial Neural Network.

        Arguments:
        layers (list): list of number of neurons in each layer
        """
        # initializing layers
        self.layers=layers

        # initializing weights
        weights=[]

        for i in range(len(layers)-1):
            w=np.random.randn(layers[i], layers[i+1])
            weights.append(w)

        self.weights=weights

        # initializing activations values with zeros
        activations = []

        for i in range(len(self.layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations=activations

        # initializing derivatives with zeros
        derivatives = []

        for i in range(len(self.layers)-1):
            d = np.zeros((self.layers[i], self.layers[i+1]))
            derivatives.append(d)

        self.derivatives=derivatives


    def _sigmoid(self, x):
        """
        Function calculates sigmoid function value based on x.

        Arguments:
        x (array): value for which sigmoid will be calculated
        """
        return 1/(1+np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        Function calculates derivative of sigmoid function value based on x.

        Arguments:
        x (double): value for which derivative of sigmoid will be calculated
        """
        return x*(1-x)

    def predict(self, X):
        """
        Function predicts output based on the input values.

        Arguments:
        X (array): input for one of training samples
        """
        # first activation values are input values
        activations=X
        # setting values of activations
        self.activations[0]=X

        for i, w in enumerate(self.weights):
            # calculating net inputs
            net_inputs=np.dot(activations, w)

            # calculating activation
            activations=self._sigmoid(net_inputs)
            self.activations[i+1]=activations
        
        # output
        return activations

    def backpropagate(self, error):
        """
        Running backpropagation.

        Arguments:
        error (array): difference between target and output

        1. Calculates delta which is current error times derivative of activation function. 
        2. Restructures delta and activation values.
        3. Calculates new derivatives as dot product of activation values and delta.
        4. Updates error for next iteration.

        dE/dW[i] = error[i] * (sigmoid'(v[i+1]))*a[i]
        """

        for i in reversed(range(len(self.derivatives))):
            # calculating error[i] * (sigmoid'(h[i+1]))
            delta=error*self._sigmoid_derivative(self.activations[i+1])

            # in order to multiply delta*a[i] reshaping of arrays are needed
            delta_reshaped=delta.reshape(delta.shape[0], -1).T
            current_activations_reshaped=self.activations[i].reshape(self.activations[i].shape[0], -1)

            # calculating delta*a[i]
            self.derivatives[i]=np.dot(current_activations_reshaped, delta_reshaped)

            # calculating new error
            error=np.dot(delta, self.weights[i].T)

    def gradient_descent(self, learning_rate):
        """
        Function performs gradient descent.

        Arguments:
        learning rate [double]: specifying how quickly net will learn
        """

        for i in range(len(self.weights)):
            self.weights[i]+=learning_rate*self.derivatives[i]

    def _mse(self, target, output):
        """
        Calculating Mean Squared Error.

        Arguments:
        target (array): target value
        output (array): value calculated by predict method
        """
        return np.average((target-output)**2)

    def fit(self, X, y, epochs, learning_rate):
        """
        Function trains neural network.

        Arguments:
        X (array2D): inputs for train set
        y (array): targets for train set
        epochs [int]: number of epochs
        learning_rate [double]: specifying how quickly net will learn

        For each epoch and for each sample it:
        1. Predicts outcome based on current weigths
        2. Calculates error
        3. Backpropagates error
        4. Performes gradient descent
        """

        for i in range(epochs):
            sum_mse=0

            for input, target in zip(X, y):
                # perform forward propagation to predict target
                output=self.predict(input)

                # calculate error
                error=target-output  

                # backpropagation
                self.backpropagate(error)

                # gradient descent
                self.gradient_descent(learning_rate)

                # calculate sum of mse
                sum_mse+=self._mse(target, output)
            
            # print mse on each epoch
            print("MSE: {} on epoch {}".format(sum_mse/len(X), i))

    
    def evaluate(self, X_test, y_test):
        outputs=self.predict(X_test)

        sum_mse=0
        for target, output in zip(y_test, outputs):
            sum_mse+=self._mse(target, output)
        
        return sum_mse/len(X_test)