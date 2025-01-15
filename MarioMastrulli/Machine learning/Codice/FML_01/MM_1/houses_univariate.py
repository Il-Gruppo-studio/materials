import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from Linear_regression import LinearRegression

case = pd.read_csv('houses_portaland_simple.csv')
print(case.describe())

case.drop("Bedroom",axis=1,inplace=True)
#inplace modifica il file direttamente senza modificarlo

plt.plot(case.Size, case.Price, 'r')
plt.show()

#ho usato il tag Tkagg non so se sia giusto ma con quello del prof mi da errore
print(case.corr())
case = case.values
#rilascia array
print(case)

media = case.mean(axis=0)
#fa la media aritmetica
std = case.std(axis=0)
#fa la deviazione standard

#ora applichiamo la standardizzazione dei dati
case = (case -media)  / std

train_index = round(len(case)* 0.8)
# indice traning set su 100 righe calcoliamo 80

x = case[:,0]
y = case[:,1]

print(x)

x = np.c_[np.ones(x.shape[0]),x]
#il c_ unisce i due array di colonne formando una matrice
#aggiunge il bias alla prima colonna y = X theta
#x_shape come se fosse la lunghezza resistuisce il numero di righe x
print(x)

#ora si passa alla regressione lineare
linear = LinearRegression(feature=x.shape[1], n_steps=1000,learning_rate=0.01)
#perche x.shape[1] ? perche è diventata una matrice.

linex = np.linspace(x[:,1].min(), x[:,1].max(),100)
#Questa riga genera un array di 100 valori equidistanti tra il minimo e il massimo della seconda colonna di x
#crea valori equidistanti tra loro
print(linex)
liney = []  # Crea una lista vuota per salvare i valori y della linea di regressione
for xx in linex:  # Per ogni valore xx (cioè un punto x) nella lista lineX
    liney.append(linear.theta[0] + linear.theta[1] * xx)
    # Calcola il corrispondente valore y usando la formula della regressione:
    # y = θ₀ (intercetta) + θ₁ (pendenza) * x
    # Aggiungi il valore calcolato alla lista liney

# Grafico dei dati di training come punti rossi
plt.plot(x[:, 1], y, 'r.', label='Training data')

# Linea di regressione calcolata (ipotesi del modello)
plt.plot(linex, liney, 'b--', label='Current hypothesis')

# Aggiunge la legenda per identificare i dati e la linea
plt.legend()

# Mostra il grafico
plt.show()

theta0_vals = np.linspace(-2,2,100)
#crea 100 valori compresi da -2 e 2
theta1_vals = np.linspace((-2,3,100))
#la scelta di 2 3 è proprio perche i nostri valori x sono in quel range

# ora dobbiamo inzializzare la matrice che contiene i valori J
j_vals = np.zeros(theta0_vals.size,theta1_vals.size)

#ora inseriamo i valori nella matrice

for t1, element in enumerate(theta0_vals):  # Ciclo sui valori di theta₀ (intercetta)
    for t2, element2 in enumerate(theta1_vals):  # Ciclo sui valori di theta₁ (pendenza)

        thetaT = np.zeros(shape=(2, 1))  # Crea una matrice 2x1 di zeri per memorizzare i parametri theta₀ e theta₁

        thetaT[0][0] = element  # Assegna il valore corrente di theta₀ alla prima riga di thetaT
        thetaT[1][0] = element2  # Assegna il valore corrente di theta₁ alla seconda riga di thetaT

        h = x.dot(thetaT.flatten())
        # Calcola l'ipotesi (h) moltiplicando la matrice delle caratteristiche (x) per i parametri theta
        # (flatten è usato per trasformare thetaT in un array 1D)

        j = (h - y)
        # Calcola l'errore tra l'ipotesi h e i valori reali y (h - y)

        J = j.dot(j) / 2 / len(x)
        # Calcola la funzione di costo (J). j.dot(j) somma i quadrati degli errori, /2 per la normalizzazione, /len(x) per ottenere la media.

        j_vals[t1, t2] = J  # Memorizza il valore della funzione di costo J nella matrice j_vals alla posizione corrispondente alla combinazione di theta₀ e theta₁




