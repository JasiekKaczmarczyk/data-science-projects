import numpy as np

class Loss:
    def calculate_loss(y_pred, y_true):
        pass

    def calculate_loss_derivative( y_pred, y_true):
        pass

class MSELoss(Loss):
    """
    Calculates MSE Loss and it's derivative
    """

    def calculate_loss(y_pred, y_true):
        return np.mean((y_pred-y_true)**2)

    def calculate_loss_derivative(y_pred, y_true):
        return 2*np.mean((y_pred-y_true), axis=0)/len(y_true)