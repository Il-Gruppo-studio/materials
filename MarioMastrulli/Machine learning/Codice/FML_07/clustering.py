
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kmeans import KMeans
from kmedoids import KMedoids


proc_data, y_true = datasets.make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=88 # For reproducibility
)

k_s = 4

mean = np.mean(proc_data,  axis=0)
std = np.std(proc_data,  axis=0)
proc_data = (proc_data - mean) / std

# K-Means from scratch
kmeans_obj_2 = KMeans(n_clusters=k_s, random_state=42)
kmeans_obj_2.fit(proc_data)
y_pred = kmeans_obj_2.predict(proc_data)
print(f"Silhouette Coefficient Home K-Means:\t{silhouette_score(proc_data,y_pred)}")

plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("K-Means")
plt.show()

#################################################################################################
# K-Medoids from scratch
kmedoids_obj = KMedoids(n_clusters=k_s, random_state=46)
kmedoids_obj.fit(proc_data)
y_pred = kmedoids_obj.predict(proc_data)
print(f"Silhouette Coefficient Home K-Medoids:\t{silhouette_score(proc_data,y_pred)}")

plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("K-Medoids")
plt.show()

plt.show()