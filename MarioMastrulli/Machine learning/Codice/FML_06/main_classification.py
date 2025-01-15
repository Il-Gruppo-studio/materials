import numpy as np
import pandas as pd
from classification_nn import NeuralNetwork

df = pd.read_csv('datasets/diabetes.csv')
print(df.describe())

features_names, label_name = df.columns[:-1], df.columns[-1]

df = df.sample(frac=1).reset_index(drop=True)

x = df[features_names].values
y = df[label_name].values

train_index = round(len(x) * 0.8)

x_train = x[:train_index]
y_train = y[:train_index]
x_test = x[train_index:]
y_test = y[train_index:]

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


nn = NeuralNetwork(layers=[x.shape[1], 10, 10, 1], lmd=0.01)
nn.fit(x_train, y_train)
print(nn.compute_performance(x_test, y_test))
