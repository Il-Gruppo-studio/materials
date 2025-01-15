import pandas as pd
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

# read the dataset of houses prices
houses = pd.read_csv('datasets/houses_portaland_simple.csv')

# print dataset stats
print(houses.describe())
houses.drop('Bedroom', axis=1, inplace=True)

# shuffling all the samples to avoid group bias
houses = houses.sample(frac=1, random_state=42).reset_index(drop=True)

plt.plot(houses.Size, houses.Price, 'r.')
plt.show()

# another way to test the correlation
print(houses.corr())

houses = houses.values

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = houses.mean(axis=0)
std = houses.std(axis=0)

# apply mean and std (standard deviation) computed on the training sample to both training and test sets
houses = (houses - mean) / std

# in order to perform hold-out, split 80/20 the samples set
train_index = round(len(houses) * 0.8)

x = houses[:, 0]
y = houses[:, 1]

plt.plot(x, y, 'r.')
plt.show()

# add bias column
x = np.c_[np.ones(x.shape[0]), x]

# create a regressor with specific characteristics
linear = LinearRegression(n_features=x.shape[1], n_steps=1000, learning_rate=0.01)

lineX = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
liney = [linear.theta[0] + linear.theta[1]*xx for xx in lineX]

plt.plot(x[:, 1], y, 'r.', label='Training data')
plt.plot(lineX, liney, 'b--', label='Current hypothesis')
plt.legend()
plt.show()

# fit (try different strategies) your trained regressor
cost_history, theta_history = linear.fit(x, y)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost:  {cost_history[-1]:.3f}''')

lineX = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
liney = [theta_history[-1, 0] + theta_history[-1, 1]*xx for xx in lineX]

plt.plot(x[:, 1], y, 'r.', label='Training data')
plt.plot(lineX, liney, 'b--', label='Current hypothesis')
plt.legend()
plt.show()

plt.plot(cost_history, 'g--')
plt.show()

#Grid over which we will calculate J
theta0_vals = np.linspace(-2, 2, 100)
theta1_vals = np.linspace(-2, 3, 100)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = np.zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        h = x.dot(thetaT.flatten())
        j = (h - y)
        J = j.dot(j) / 2 / (len(x))
        J_vals[t1, t2] = J

#Contour plot
J_vals = J_vals.T

A, B = np.meshgrid(theta0_vals, theta1_vals)
C = J_vals

cp = plt.contourf(A, B, C)
plt.colorbar(cp)
plt.plot(theta_history.T[0], theta_history.T[1], 'r--')
plt.show()