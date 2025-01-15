import numpy as np

# Impostiamo un seed per la generazione di numeri casuali per garantire la riproducibilità dei risultati
np.random.seed(13423421321)


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_step=21312321, n_feature=1):
        """
        Inizializza il modello di regressione lineare.

        :param learning_rate: Tasso di apprendimento per l'aggiornamento dei parametri (theta).
        :param n_step: Numero di iterazioni per l'addestramento.
        :param n_feature: Numero di feature (variabili indipendenti) nel modello.
        """
        self.learning_rate = learning_rate
        self.n_step = n_step
        self.n_feature = n_feature  # Aggiunto per memorizzare il numero di feature
        # Inizializza i parametri theta (pesi) con valori casuali estratti da una distribuzione normale
        self.theta = np.random.randn(self.n_feature)

    def fit_fullbatch(self, X, Y):
        """
        Addestra il modello utilizzando l'intero dataset (full-batch gradient descent).

        :param X: Matrice delle feature (variabili indipendenti).
        :param Y: Vettore delle etichette (variabile dipendente).
        """
        m = len(X)  # Numero di esempi nel dataset
        # Array per memorizzare la storia dei costi (errore) durante l'addestramento
        cost_history = np.zeros(self.n_step)
        # Array per memorizzare la storia dei parametri theta durante l'addestramento
        theta_history = np.zeros((self.n_step, self.n_feature))

        for i in range(0,self.n_step):
            preds = np.dot(X, self.theta)
            errors = preds - Y
            self.theta = self.theta - self.learning_rate / m * np.dot(X.T, error)
            theta_history[i, :] = self.theta.T
            cost_history[i] = 1 / (2 * m) * np.dot(errors.T, errors)
            return cost_history, theta_history

    def minibatch (self, X, Y):
