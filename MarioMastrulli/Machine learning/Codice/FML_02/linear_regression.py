import numpy as np

from FML_01.houses_multivariate import X_train

np.random.seed(123)

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_steps=200, n_features=1):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.randn(n_features)

    def fit_fullbatch(self, X, y):
        m=len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds= np.dot(X, self.theta)
            error = preds-y

            self.theta = self.theta - self.learning_rate/m * np.dot(X.T, error)
            theta_history[step, :] = self.theta.T
            cost_history[step]= 1/(2*m)*np.dot(error.T,error)
            return cost_history, theta_history


    def fit_minibatch(self, X_train, y_train, batch_size=4):
        mt= X_train.shape[0]
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for epoch in range(self.n_steps):
            for index in range (0, mt, batch_size):
                x_i = X_train[index:index+batch_size]
                y_i = y_train[index:index+batch_size]

                pred_i = np.dot(x_i, self.theta)
                error_i= pred_i - y_i

            self.theta = self.theta-(self.learning_rate/batch_size)* np.dot(x_i.T, error_i)
            pred_train = np.dot(X_train, self.theta)
            error_train = pred_train - y_train
            cost_history[epoch]= (1/(2*mt)* np.dot(error_train.T, error_train))
            theta_history[epoch, :] = self.theta.T

        return cost_history, theta_history


    def fit_sgd(self, X, y):
        m=len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for epoch in range(self.n_steps):
            random_index = np.random.randint(m)
            x_i=X[random_index]
            y_i=y[random_index]
            prediction = np.dot(x_i, self.theta)
            error = prediction - y_i
            self.theta = self.theta - self.learning_rate * x_i.T * error
            theta_history[epoch, :] = self.theta.T
            prediction = np.dot(X, self.theta)
            cost = (1/(2*m))*np.sum(prediction-y)**2
            cost_history[epoch] = cost
        return cost_history, theta_history


    def predict(self, X):
        return np.dot(X, self.theta)
