import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, n_features, n_steps):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.randn(n_features)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit_full_batch(self, x, y):
        m = len(x)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(self.n_steps):
            z = np.dot(x, self.theta)
            prediction = self.sigmoid(z)
            error = prediction - y

            self.theta = self.theta - self.learning_rate / m * np.dot(x.T, error)

            cost_history[step] = - (1/m) * (np.dot(y, np.log(prediction)) + np.dot(1-y, np.log(1-prediction)))
            theta_history[step, :] = self.theta

        return cost_history, theta_history

    def predict(self, x, threshold=0.5):
        z = np.dot(x, self.theta)
        prediction = self.sigmoid(z)
        return prediction > threshold




