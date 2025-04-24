import numpy as np

np.random.seed(123)

class LinearRegression:
    """
    Class to model a linear regression. This class has all the methods to be trained with different strategies
    and one method to produce a full prediction based on input samples. Moreover, this one is equipped by one method to
    measure performance and another method to build learning curves
    """
    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1):
        """
        :param learning_rate: learning rate value
        :param n_steps: number of epochs for the training
        :param n_features: number of features involved in the regression
        :param lmd: regularization factor

        lmd_ is an array useful when is necessary compute theta's update with regularization factor
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)

    def fit(self, X, y):
        """
        apply full batch gradient descent, without regularization, to the training set and return the evolution
        history of train and validation costs.
        :param X: training samples with bias
        :param y: training target values
        :param X_test: validation samples with bias
        :param y_test: validation target values
        :return: history of evolution about cost and theta during training steps and, cost during validation phase
        """
        m = len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X, self.theta)

            error = preds - y

            self.theta = self.theta - self.learning_rate / m * np.dot(X.T, error)
            theta_history[step, :] = self.theta.T
            cost_history[step] = 1 / (2 * m) * np.dot(error.T, error)

        return cost_history, theta_history

    def predict(self, X):
        """
        perform a complete prediction on X samples
        :param X: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m,)
        """
        return np.dot(X, self.theta)