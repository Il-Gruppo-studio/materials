from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMedoids(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        """
        Initializes the KMedoids clustering instance.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - dist (function): The distance function to use. Defaults to Euclidean distance.
        - random_state (int): Seed for random number generator for reproducibility.
        """
        self.n_clusters = n_clusters  # Number of clusters to form
        self.dist = dist  # Distance function to compute distances
        self.rstate = np.random.RandomState(random_state)  # Random state for reproducibility
        self.cluster_centers_ = []  # List to store medoids (centroid-like representatives)
        self.indices = []  # List to store indices of the current medoids

    def fit(self, X):
        """
        Computes K-Medoids clustering on the data X.

        Parameters:
        - X (numpy.ndarray): The input data, shape (n_samples, n_features)
        """
        # Shortcut to the randint method of the RandomState instance for selecting random indices
        rint = self.rstate.randint

        # Step 1: Initialize medoids by randomly selecting K unique data point indices from X
        self.indices = [rint(X.shape[0])]  # Randomly select the first medoid index
        for _ in range(self.n_clusters - 1):
            i = rint(X.shape[0])
            # Ensure that each selected index is unique to avoid duplicate medoids
            while i in self.indices:
                i = rint(X.shape[0])
            self.indices.append(i)
        # Initialize medoids with the selected data points
        self.cluster_centers_ = X[self.indices, :]

        # Assign each data point to the nearest medoid
        self.y_pred = np.argmin(self.dist(X, self.cluster_centers_), axis=1)

        # Compute the initial cost (total distance within clusters)
        cost, _ = self.compute_cost(X, self.indices)
        new_cost = cost  # Initialize new_cost with the initial cost
        new_y_pred = self.y_pred.copy()  # Copy of current cluster assignments
        new_indices = self.indices[:]  # Copy of current medoid indices
        initial = True  # Flag to ensure at least one iteration

        # Step 2: Iteratively improve medoid positions to minimize total cost
        while (new_cost < cost) | initial:
            initial = False  # After the first iteration, initialization is done
            cost = new_cost  # Update the cost to the new_cost
            self.y_pred = new_y_pred  # Update cluster assignments
            self.indices = new_indices  # Update medoid indices

            # Iterate over each cluster to find potential better medoids
            for k in range(self.n_clusters):
                # Iterate over all data points assigned to cluster k
                for r in [i for i, x in enumerate(new_y_pred == k) if x]:
                    if r not in self.indices:
                        # Create a temporary copy of current medoid indices
                        indices_temp = self.indices[:]
                        # Replace the k-th medoid with the current point r
                        indices_temp[k] = r
                        # Compute the cost for this new set of medoids
                        new_cost_temp, y_pred_temp = self.compute_cost(X, indices_temp)
                        # If the new cost is better (lower), update the best found so far
                        if new_cost_temp < new_cost:
                            new_cost = new_cost_temp  # Update the new cost
                            new_y_pred = y_pred_temp  # Update cluster assignments
                            new_indices = indices_temp  # Update medoid indices

        # After convergence, update the medoid coordinates based on the final indices
        self.cluster_centers_ = X[self.indices, :]

    def compute_cost(self, X, indices):
        """
        Computes the total cost (sum of distances) for the current set of medoids.

        Parameters:
        - X (numpy.ndarray): The input data, shape (n_samples, n_features)
        - indices (list): List of indices representing the current medoids

        Returns:
        - total_cost (float): The total sum of distances within all clusters
        - y_pred (numpy.ndarray): Array of cluster assignments for each data point
        """
        # Assign each data point to the nearest medoid
        y_pred = np.argmin(self.dist(X, X[indices, :]), axis=1)

        # Calculate the total cost by summing distances of points to their assigned medoids
        total_cost = np.sum([
            np.sum(self.dist(X[y_pred == i], X[[indices[i]], :]))
            for i in set(y_pred)
        ])

        return total_cost, y_pred

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to.

        Parameters:
        - X (numpy.ndarray): New data to predict, shape (n_samples, n_features)

        Returns:
        - y_pred (numpy.ndarray): Index of the cluster each sample belongs to, shape (n_samples,)
        """
        # Compute distances between X and the current medoids
        distances = self.dist(X, self.cluster_centers_)
        # Assign each data point to the nearest medoid
        y_pred = np.argmin(distances, axis=1)
        return y_pred