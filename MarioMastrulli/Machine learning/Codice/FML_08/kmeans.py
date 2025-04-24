from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMeans(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        """
        Initializes the KMeans clustering instance.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - dist (function): The distance function to use. Defaults to Euclidean distance.
        - random_state (int): Seed for random number generator for reproducibility.
        """
        self.n_clusters = n_clusters  # Number of clusters to form
        self.dist = dist  # Distance function to compute distances
        self.rstate = np.random.RandomState(random_state)  # Random state for reproducibility
        self.cluster_centers_ = []  # List to store centroids of clusters

    def fit(self, X):
        """
        Computes K-Means clustering on the data X.

        Parameters:
        - X (numpy.ndarray): The input data, shape (n_samples, n_features)
        """
        # Shortcut to the randint method of the RandomState instance for selecting random indices
        rint = self.rstate.randint

        # Step 1: Initialize centroids by randomly selecting K unique data points from X
        initial_indices = [rint(X.shape[0])]  # Randomly select the first centroid index
        for _ in range(self.n_clusters - 1):
            i = rint(X.shape[0])
            # Ensure that each selected index is unique to avoid duplicate centroids
            while i in initial_indices:
                i = rint(X.shape[0])
            initial_indices.append(i)
        # Initialize cluster centers with the selected data points
        self.cluster_centers_ = X[initial_indices, :]

        # Flag to control the convergence of the algorithm
        continue_condition = True

        # Step 2: Repeat the process until convergence (i.e., centroids do not change)
        while continue_condition:
            # Store old centroids to check for convergence later
            old_centroids = self.cluster_centers_.copy()

            # Step 3: Assign each data point to the nearest centroid
            # Compute the distance from each point to each centroid
            # self.dist returns a distance matrix of shape (n_samples, n_clusters)
            self.y_pred = np.argmin(self.dist(X, self.cluster_centers_), axis=1)
            # self.y_pred now contains the index of the closest centroid for each data point

            # Step 4: Update centroids by computing the mean of all points assigned to each cluster
            for i in set(self.y_pred):
                # Select all data points assigned to cluster i
                cluster_points = X[self.y_pred == i]
                # Compute the new centroid as the mean of these points
                self.cluster_centers_[i] = np.mean(cluster_points, axis=0)

            # Step 5: Check for convergence
            # If centroids have not changed after the update, the algorithm has converged
            if (old_centroids == self.cluster_centers_).all():
                continue_condition = False  # Exit the loop
                # If centroids have changed, the loop continues for another iteration

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to.

        Parameters:
        - X (numpy.ndarray): New data to predict, shape (n_samples, n_features)

        Returns:
        - y_pred (numpy.ndarray): Index of the cluster each sample belongs to, shape (n_samples,)
        """
        # Compute distances between X and the current centroids
        distances = self.dist(X, self.cluster_centers_)
        # Assign each data point to the nearest centroid
        y_pred = np.argmin(distances, axis=1)
        return y_pred