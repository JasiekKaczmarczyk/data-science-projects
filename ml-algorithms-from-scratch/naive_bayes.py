import numpy as np

class NaiveBayes:
    def fit(self, X_train, y_train):
        """
        Fits training data to model

        Arguments:
        X_train (array): training set features
        y_train (array): training set labels
        """
        # number of features
        n_samples, n_features = X_train.shape

        # unique classes
        self.labels=np.unique(y_train)
        n_labels=len(self.labels)

        # initializing mean, variance and a prori values for each label
        self.mean=np.zeros((n_labels, n_features))
        self.variance=np.zeros((n_labels, n_features))
        self.aprioris=np.zeros(n_labels)

        for label in self.labels:
            # grabbing samples from X that are of certain label
            X_with_label=X_train[label==y_train]

            # calculate mean, variance and a prori values
            self.mean[label, :]=np.mean(X_with_label, axis=0)
            self.variance[label, :]=np.var(X_with_label, axis=0)
            self.aprioris[label]=len(X_with_label)/n_samples


    def predict(self, X):
        """
        Predicts labels to each sample in array

        Arguments:
        X (array): data on which we predict labels
        """
        return [self._predict_label(x) for x in X]
    
    def _predict_label(self, x):
        """
        Predicts label to single sample

        P(y|X) = log(P(x1|y)) + ... + log(P(xn|y)) + log(P(y))
        """
        # list of a posteriori values for each label
        aposterioris = []

        for i in range(len(self.labels)):
            # calculate log of a priori
            apriori=np.log(self.aprioris[i])
            # calculate log of conditional probabilities log(P(x1|y)) + ... + log(P(xn|y)) based on standard normal distribution
            conditional_probability=np.log(self._normal_distribution(x, i))
            # sum values
            aposteriori=np.sum(conditional_probability) + apriori
            aposterioris.append(aposteriori)
        
        # return label that has the highest likelihood
        return self.labels[np.argmax(aposterioris)]

    def _normal_distribution(self, x, i):
        """
        Calculates value for x for certain label based on normal distribution
        """
        mean=self.mean[i]
        variance=self.variance[i]
        return np.exp(-(x-mean)**2/(2*variance))/(2*np.pi*variance)
