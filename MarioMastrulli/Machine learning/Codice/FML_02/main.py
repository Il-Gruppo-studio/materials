import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from evaluation import Evaluation
house= pd.read_csv('../houses.csv')

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

X_train = np.c_[np.ones(X_train.shape[0]),X_train]
X_test = np.c_[np.ones(X_test.shape[0]),X_test]

linear = LinearRegression(n_features=X_train.shape[1], n_steps=100, learning_rate=0.1)

cost_history, theta_history = linear.fit_sgd(X_train, y_train)

print(f'''Thetas: {*linear.theta,})''')
print(f''' Final Train cost: {cost_history[-1]: 3f}''')

plt.plot(cost_history, 'g--')
plt.show()

eval = Evaluation(linear)
print(eval.compute_performance(X_test, y_test))



