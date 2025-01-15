import pandas as pd
import numpy as np

house = pd.read_csv('datasets/houses.csv')

print(house.describe())

house = house.sample(frac=1).reset_index(drop=True)

x = house[['GrLivArea','LotArea','GarageArea','FullBath']].values

y = house['SalePrice'].values

train_index = round(0.8 * len(x))

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

# compute theta following the normal equation formula

theta = np.dot(np.dot(np.linalg.inv(np.dot(x_train.T, x_train)), x_train.T), y_train)

# estimate values on the test set
y_test_hat = np.dot(x_test, theta)

# try to evaluate all the metrics by yourself

