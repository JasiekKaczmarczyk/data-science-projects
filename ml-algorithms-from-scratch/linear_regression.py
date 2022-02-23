import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        """
        Constructor for Linear Regression Model

        Arguments:
        learning_rate [double]: tells how fast algorithm learns
        n_iterations [int]: number of iterations of gradient descent
        """
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.slope=None
        self.bias=None

    def fit(self, X_train, y_train):
        """
        Fits training data to model

        Arguments:
        X_train (array): training set features
        y_train (array): training set labels
        """
        # number of features
        n_features = X_train.shape[1]

        # initializing slope and bias values
        self.slope=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iterations):
            # gradient descent
            self.slope-=self.learning_rate*self._slope_derivative(X_train, y_train)
            self.bias-=self.learning_rate*self._bias_derivative(X_train, y_train)

    def _regression_line(self, X):
        """
        Calculates values for regression line
        """
        return np.dot(X, self.slope) + self.bias

    def _slope_derivative(self, X, y):
        """
        Calculates derivative for slope
        """
        return np.dot(X.T, (self._regression_line(X) - y)) / len(X)

    def _bias_derivative(self, X, y):
        """
        Calculates derivative for bias
        """
        return np.sum((self._regression_line(X) - y)) / len(X)

    def predict(self, X):
        """
        Predicts values to each sample in array

        Arguments:
        X (array): data on which we predict values
        """
        return self._regression_line(X)

    