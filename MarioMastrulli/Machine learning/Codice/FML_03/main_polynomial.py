import pandas as pd
import numpy as np
from FML2.linear_regression import LinearRegression
from evaluation import Evaluation
from utilities import plot_theta_gd

# read the dataset of houses prices
salary = pd.read_csv('datasets/Position_Salaries_base.csv')
salary = salary.sample(frac=1).reset_index(drop=True)

print(salary.columns)

salary = salary.values

train_index = round(len(salary) * 0.8)

x = salary[:, 0]
y = salary[:, 1]

x_train = x[:train_index]
y_train = y[:train_index]
x_test = x[train_index:]
y_test = y[train_index:]

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

x_train_squared = x_train ** 2
x_test_squared = x_test ** 2

x_train_cubic = x_train ** 3
x_test_cubic = x_test ** 3

x_train = np.column_stack((x_train, x_train_squared, x_train_cubic))
x_test = np.column_stack((x_test, x_test_squared, x_test_cubic))

x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]

linear = LinearRegression(learning_rate=0.01, n_steps=1000, n_features=x_train.shape[1])

cost_history, theta_history = linear.fit_fullbatch(x_train, y_train)
print(cost_history[-1])

eval = Evaluation(linear)
print(eval.compute_performance(x_test, y_test))

plot_theta_gd(x_train, y_train, linear, cost_history, theta_history, 0, 1)
