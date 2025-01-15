import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Linear_regression import LinearRegression


houses = pd.read_csv('../datasets/houses.csv')
print(houses.describe())
# statistiche generali

houses = houses.sample(frac=1).reset_index(drop=True)
#campiona tutte le righe FRAC=1 100% delle righe e viene re-indicizzato per evitare
#il bias di gruppo

print(" Le feature prese sono rispettivamente: ")
x = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values
y = houses['SalePrice'].values

#PARTE DI TRAINING
train_index = round(len(x)*0.8)
#tipo su 100 train_index vale 80

# Training set: selezione delle prime `train_index` righe per le feature (X_train) e il target (Y_train)
X_train = x[:train_index]  # Prende le prime `train_index` righe di `x` (feature) per il training set
Y_train = y[:train_index]  # Prende le prime `train_index` righe di `y` (target) per il training set
# ":" è UN ESTREMITA SOPRA INDICA DALL'INZIO FINO A TRAIN INDEX

# Test set: selezione delle righe da `train_index` in poi per le feature (X_test) e il target (Y_test)
X_test = x[train_index:]   # Prende le righe da `train_index` in poi di `x` (feature) per il test set
Y_test = y[train_index:]# Prende le righe da `train_index` in poi di `y` (target) per il test set

#suddivisione dataset

#calcoliamo ora la media e deviazione standard
media = X_train.mean(axis=0)
dev_std = X_train.std(axis=0)
#axis per ogni colonna della matrice, viene calcolata la media di tutti i valori in quella colonna.

#ora bisogna appunto normalizzare e lo si fa cosi:
X_train = (X_train - media) / dev_std
X_test = (X_test - media) / dev_std

#SIA PER L'ALLENAMENTO CHE PER IL TEST

#aggiungere il bias theta 0

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0],X_test)]
# SI LEGGE CREA UNA COLONNA DI 1 e la aggiunge alla priam colonne DI X_train e test

linear = LinearRegression(feature=X_train.shape[1], n_steps=1000,learning_rate=0.05)
cost_history = linear.fit(X_train, Y_train)
theta_history = linear.fit(X_train, Y_train)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost:  {cost_history[-1]:.3f}''')

plt.plot(cost_history, 'g--')
plt.show()
