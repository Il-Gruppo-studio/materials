import pandas as pd
import numpy as np

from utilities import plot_theta_gd
from evaluation import Evaluation
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

# read the dataset of houses prices
houses = pd.read_csv('datasets/houses.csv')

# print dataset stats
print(houses.describe())

# shuffling all samples to avoid group bias
houses = houses.sample(frac=1).reset_index(drop=True)

# select only some features, also you can try with other features
x = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values

# select target value
y = houses['SalePrice'].values

# in order to perform hold-out splitting 80/20 identify max train index value
train_index = round(len(x) * 0.8)

# split dataset into training and test
X_train = x[:train_index]
y_train = y[:train_index]

X_test = x[train_index:]
y_test = y[train_index:]

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# apply mean and std (standard deviation) compute on training sample to training set and to test set
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# add bias column
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# create a regressor with specific characteristics
linear = LinearRegression(n_features=X_train.shape[1], n_steps=1000, learning_rate=0.05)

# fit (try different strategies) your trained regressor
cost_history, theta_history = linear.fit(X_train, y_train)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost:  {cost_history[-1]:.3f}''')

plt.plot(cost_history, 'g--')
plt.show()

eval = Evaluation(linear)

print(eval.compute_performance(X_test, y_test))

plot_theta_gd(X_train, y_train, linear, cost_history, theta_history, 0, 3)
