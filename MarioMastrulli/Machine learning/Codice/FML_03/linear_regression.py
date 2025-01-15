import numpy as np

np.random.seed(123)

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_steps=200, n_features=1, lmd=0.01):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.randn(n_features)
        self.lmd = lmd

        self.lmd_ = np.zeros(n_features)
        self.lmd_ = np.full(n_features, lmd)
        self.lmd_[0] = 0

    def fit_fullbatch(self, X, y):
        m=len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X, self.theta)
            error = preds-y

            self.theta = self.theta - self.learning_rate / m * np.dot(X.T, error)
            theta_history[step, :] = self.theta.T
            cost_history[step]= 1/(2*m) * np.dot(error.T,error)

        return cost_history, theta_history

    def fit_fullbatch_regularization(self, X, y):
        m=len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X, self.theta)
            error = preds-y

            self.theta = self.theta - self.learning_rate / m * (np.dot(X.T, error) + self.lmd_ * self.theta)
            theta_history[step, :] = self.theta.T
            cost_history[step]= 1/(2*m) * (np.dot(error.T,error) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:]))
        return cost_history, theta_history


    def fit_minibatch(self, X_train, y_train, batch_size=4):
        mt= X_train.shape[0]
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for epoch in range(self.n_steps):
            for index in range (0, mt, batch_size):
                x_i = X_train[index:index+batch_size]
                y_i = y_train[index:index+batch_size]

                pred_i = np.dot(x_i, self.theta)
                error_i = pred_i - y_i

                self.theta = self.theta-(self.learning_rate/batch_size) * np.dot(x_i.T, error_i)
            pred_train = np.dot(X_train, self.theta)
            error_train = pred_train - y_train
            cost_history[epoch]= (1/(2*mt)* np.dot(error_train.T, error_train))
            theta_history[epoch, :] = self.theta.T

        return cost_history, theta_history


    def fit_sgd(self, X, y):
        m=len(X)
        cost_history = np.zeros(self.n_steps*m)
        theta_history= np.zeros((self.n_steps*m, self.theta.shape[0]))

        for epoch in range(self.n_steps):
            for index in range(m):
                x_i=X[index]
                y_i=y[index]
                prediction = np.dot(x_i, self.theta)
                error = prediction - y_i
                self.theta = self.theta - self.learning_rate * x_i.T * error
                theta_history[(epoch*m)+index, :] = self.theta.T
                prediction = np.dot(X, self.theta)
                cost = (1/2)*np.sum((prediction-y)**2)
                cost_history[(epoch*m)+index] = cost
        return cost_history, theta_history


    def predict(self, X):
        return np.dot(X, self.theta)
