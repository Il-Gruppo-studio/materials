
import numpy as np
from sklearn import datasets
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kmeans import KMeans
from kmedoids import KMedoids
from sklearn.cluster import KMeans as KMeans_scikit

# Step 1: Data Preparation
proc_data, y_true = datasets.make_blobs(n_samples=500,
                                        n_features=2,
                                        centers=4,
                                        cluster_std=1,
                                        center_box=(-10.0, 10.0),
                                        shuffle=True,
                                        random_state=88)  # For reproducibility

# Normalize the data
mean = np.mean(proc_data, axis=0)
std = np.std(proc_data, axis=0)
proc_data = (proc_data - mean) / std

# Step 2: Elbow Method Implementation

# Initialize list to store WCSS for different K
wcss = []

# Define range of K values to try
K_range = range(1, 11)  # 1 to 10

for k in K_range:
    # Initialize KMeans with current K
    kmeans = KMeans(n_clusters=k, random_state=42)
    # Fit the model
    kmeans.fit(proc_data)
    # Predict cluster assignments
    y_pred = kmeans.predict(proc_data)

    # Calculate WCSS
    current_wcss = 0
    for cluster in range(k):
        # Get all points assigned to the current cluster
        cluster_points = proc_data[y_pred == cluster]
        if cluster_points.size == 0:
            continue  # Avoid division by zero
        # Calculate squared distances to the centroid
        squared_distances = np.sum((cluster_points - kmeans.cluster_centers_[cluster]) ** 2)
        # Add to current WCSS
        current_wcss += squared_distances
    # Append current WCSS to the lists
    wcss.append(current_wcss)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, 'bo-', markersize=8)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters K')
plt.ylabel('Total Within-Cluster Sum of Squares (WCSS)')
plt.xticks(K_range)
plt.grid(True)
plt.show()


##### We can also use the Scikit-Learn implementation of the Kmeans algorithm in order to calculate the intertia (WCSS) metric.

# Initialize list to store WCSS for different K
wcss = []

# Define range of K values to try
K_range = range(1, 11)  # 1 to 10

for k in K_range:
    # Initialize KMeans with current K
    kmeans_scikit = KMeans_scikit(n_clusters=k, random_state=42)
    # Fit the model
    kmeans_scikit.fit(proc_data)
    # Predict cluster assignments
    y_pred_scikit = kmeans_scikit.predict(proc_data)

    # Calculate WCSS
    current_wcss = 0
    for cluster in range(k):

        current_wcss += kmeans_scikit.inertia_
    # Append current WCSS to the lists
    wcss.append(current_wcss)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, 'bo-', markersize=8)
plt.title('Elbow Method for Optimal K using Scikit-learn Kmeans implementation')
plt.xlabel('Number of clusters K')
plt.ylabel('Total Within-Cluster Sum of Squares (inertia)')
plt.xticks(K_range)
plt.grid(True)
plt.show()


# As we can see from the plot, let's choose K=4 or K=3 as K from data generation
# We will see that with these conditions, the Silhouette score with K=4 is higher than the score with K=3.
# consider that the elbow only judges the compactness of the clusters but not the separation,
# that's why silhouette can be a better metric to consider


optimal_k = 3

# K-Means with Optimal K
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(proc_data)
y_pred_kmeans = kmeans_optimal.predict(proc_data)
print(f"Silhouette Coefficient for K-Means with K={optimal_k}:\t{silhouette_score(proc_data, y_pred_kmeans):.4f}")

# Plot K-Means Clustering Results
plt.figure(figsize=(8, 5))
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred_kmeans, s=30, cmap='viridis')
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], kmeans_optimal.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroids')
plt.title(f"K-Means Clustering with K={optimal_k}")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# K-Medoids from scratch
kmedoids_obj = KMedoids(n_clusters=optimal_k, random_state=46)
kmedoids_obj.fit(proc_data)
y_pred_kmedoids = kmedoids_obj.predict(proc_data)
print(f"Silhouette Coefficient for K-Medoids with K={optimal_k}:\t{silhouette_score(proc_data, y_pred_kmedoids):.4f}")

# Plot K-Medoids Clustering Results
plt.figure(figsize=(8, 5))
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred_kmedoids, s=30, cmap='viridis')
plt.scatter(kmedoids_obj.cluster_centers_[:, 0], kmedoids_obj.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Medoids')
plt.title(f"K-Medoids Clustering with K={optimal_k}")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#################################################################

# DBSCAN with scikit-learn
from sklearn.cluster import DBSCAN  # Import DBSCAN from scikit-learn

# Experiment with different eps and min_samples values
# eps_values = [0.2, 0.3, 0.5]
# min_samples_values = [3, 5, 10]

dbscan_obj = DBSCAN(eps=0.2, min_samples=5)
dbscan_obj.fit(proc_data)
y_pred_dbscan = dbscan_obj.labels_

# Silhouette score requires at least 2 clusters
if len(set(y_pred_dbscan)) > 1:
    dbscan_silhouette = silhouette_score(proc_data, y_pred_dbscan)
    print(f"Silhouette Coefficient for DBSCAN:\t{dbscan_silhouette:.4f}")
else:
    print("Silhouette Coefficient for DBSCAN:\tCannot compute silhouette score with only one cluster.")

plt.figure(figsize=(8, 5))
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred_dbscan, s=30, cmap='viridis')
plt.title("DBSCAN (scikit-learn)")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
