import pandas as pd
import numpy as np
from regression_nn import NeuralNetwork

house= pd.read_csv('datasets/houses.csv')

print(house.describe())

house = house.sample(frac=1).reset_index(drop=True)

x = house[['GrLivArea','LotArea','GarageArea','FullBath']].values

y = house['SalePrice'].values

train_index = round(len(x)*0.8)

X_train = x[:train_index]
y_train = y[:train_index]

X_test = x[train_index:]
y_test = y[train_index:]

mean= X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

nn = NeuralNetwork(layers=[X_train.shape[1], 10, 8, 1], lmd=0.01)
nn.fit(X_train, y_train)
print(nn.compute_performance(X_test, y_test))



