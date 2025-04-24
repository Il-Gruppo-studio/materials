import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from classification_metrics import ClassificationMetrics

df = pd.read_csv('data/cardio.csv')

zeros = df[df.y == 0]
ones = df[df.y == 1]

zeros_train, zeros_test = train_test_split(zeros, test_size=0.1)

training = zeros_train.drop(columns='y').values
test = pd.concat([zeros_test, ones]).drop(columns='y').values
labels = np.hstack([np.zeros(zeros_test.shape[0]), np.ones(ones.shape[0])])

plt.scatter(training[:, 1], training[:, 2])
plt.show()

data_complete = df.drop(columns='y').values
labels_complete = df['y'].values

# just for visualization
tsne = TSNE(n_components=2)
data_viz = tsne.fit_transform(data_complete)
plt.scatter(data_viz[:, 0], data_viz[:, 1], c=labels_complete)
plt.show()

gmm = GaussianMixture(n_components=5, covariance_type='full')
gmm.fit(training)

print(gmm.weights_.dot(gmm.predict_proba(training).T))
predictions = gmm.weights_.dot(gmm.predict_proba(test).T)

metrics = ClassificationMetrics(gmm)
print(metrics.compute_performance(test, labels, 0.1))


