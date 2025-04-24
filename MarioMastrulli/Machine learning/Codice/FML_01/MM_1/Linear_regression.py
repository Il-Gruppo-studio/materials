import pandas as pd
import numpy as np

np.random.seed(100)
# senza questo essenzialmente cambia la parte di training e di addestramento e quindi si fissa

class LinearRegression:
    def __init__(self,learning_rate=0.01,n_steps=1000,feature=1):
        """
           :param learning_rate: learning rate velocita del modello
           :param n_steps: number of epochs for the training
           :param n_features: number of features involved in the regression
           il self afficnhe gli attributi del costruttore diventano parte dell'oggetto
           e tale quindi permette di utilizzare tali parametri in altri file py
           """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self_feature = feature
        self.theta = np.random.rand(feature)

#ora dobbiamo implementare il metodo fit che serve proprio a costituire il gradient discent

    def fit(self,x,y):
        # x matrice y invece i valori reali da prevedere

        m = len(x)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps,self.theta.shape[0]))
        """theta_history = [
        [theta_0_iter1, theta_1_iter1],
        [theta_0_iter2, theta_1_iter2],
        [theta_0_iter3, theta_1_iter3],
        ...
        
        PERCHE APPUNTO 1 è il bias e l'altro e il peso 
    ]"""
        #da capire il theta shape

        for step in range(0,self.n_steps):
            #esegue l'iterazione da 0 a n iterazioni

            previsioni = np.dot(x,self.theta)
            # fa il prodotto scalare

            errore = previsioni - y #valore reale
            #applica la formuletta
            #aggiorna i parametri per ridurre l'errore ( FORMULETTA )
            self.theta = self.theta - self.learning_rate / m * np.dot(x.T, errore)
            theta_history[step, :] = self.theta.T
            #Significa: "Prendi la riga step di theta_history e tutte le sue colonne."
            #eleziona tutte le colonne di una specifica riga (nel nostro caso, tutte le colonne di theta history_step
            cost_history = 1/(2*m)*np.dot(errore.T, errore)
            """Misura del progresso:
    
    Calcolando J(θ) a ogni iterazione, puoi vedere se il costo sta diminuendo.
    Un costo in diminuzione significa che il modello sta imparando a fare previsioni migliori."""

            return cost_history,theta_history

    def predict(self, X):
        """
        perform a complete prediction on X samples
        :param X: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m)
        """
        return np.dot(X, self.theta)

"""Dopo aver addestrato il modello con fit, il metodo predict utilizza i parametri 
θ appresi per calcolare previsioni su nuovi dati 𝑋
Output semplice:

Restituisce un array con le previsioni corrispondenti a ciascun campione."""