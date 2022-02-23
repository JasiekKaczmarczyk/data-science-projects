import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        """
        Constructor for K-Nearest Neighbour

        Arguments:
        k [int]: parameter which tells how many neighbours are taken into account during classification
        """
        self.k=k

    def fit(self, X_train, y_train):
        """
        Fits training data to classifier

        Arguments:
        X_train (array): training set features
        y_train (array): training set labels
        """
        self.X_train=X_train
        self.y_train=y_train
    
    def predict(self, X):
        """
        Predicts labels to each sample in array

        Arguments:
        X (array): data on which we predict labels
        """
        return np.array([self._predict_label(x) for x in X])

    def _predict_label(self, x):
        """
        Predicts label to single sample
        """
        # calculate distances
        distances=[self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # select k nearest
        k_idx=np.argsort(distances)[:self.k]
        labels=self.y_train[k_idx]

        # majority vote
        return Counter(labels).most_common(1)[0][0]

    
    def _euclidean_distance(self, x1, x2):
        """
        Calculates euclidean distance
        """
        return np.sqrt(np.sum((x1-x2)**2))