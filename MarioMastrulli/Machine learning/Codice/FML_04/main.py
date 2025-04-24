import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression
from utilities import plot_theta_gd

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

x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]

logistic = LogisticRegression(learning_rate=0.005, n_steps=1000, n_features=x_train.shape[1])
cost_history, theta_history = logistic.fit_full_batch(x_train, y_train)
plot_theta_gd(x_train, y_train, logistic, cost_history, theta_history, 0, 1)