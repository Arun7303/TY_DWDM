import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def initialize_centroids(X, k):
    """Randomly initialize the centroids."""
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    return centroids

def assign_clusters(X, centroids):
    """Assign each point in X to the nearest centroid."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def compute_centroids(X, labels, k):
    """Compute new centroids as the mean of assigned points."""
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def k_means(X, k, max_iters=100):
    """Perform k-means clustering."""
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = compute_centroids(X, labels, k)
        
        # If the centroids do not change, we have converged
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

def calculate_wcss(X, centroids, labels):
    """Calculate the within-cluster sum of squares (WCSS)."""
    wcss = sum(np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(len(centroids)))
    return wcss

def plot_elbow_method(X, max_k=10):
    """Plot the elbow method graph for choosing k."""
    wcss = []
    
    # Calculate WCSS for each value of k
    for k in range(1, max_k + 1):
        centroids, labels = k_means(X, k)
        wcss.append(calculate_wcss(X, centroids, labels))
    
    # Plot WCSS vs. number of clusters
    plt.figure()
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.show()

def plot_clusters(X, centroids, labels, k):
    """Plot the data points and centroids (for the first two dimensions)."""
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for i in range(k):
        points = X[labels == i]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=f'Cluster {i+1}')
    
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='k', marker='X', label='Centroids')
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Means Clustering")
    plt.show()

# Choose whether to load example dataset or custom CSV dataset
use_iris_example = True  # Set to False to use your own CSV file

if use_iris_example:
    # Load and preprocess the Iris dataset
    data = load_iris()
    X = data.data  # use all four features for clustering
else:
    # Load your own dataset
    df = pd.read_csv('Crop_yeild.csv')  # Replace with your CSV file path
    X = df.values  # assuming all columns are features

# Standardize the data for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot the elbow method to find optimal k
plot_elbow_method(X_scaled, max_k=10)

# Perform k-means clustering with a chosen k (e.g., k=3 based on elbow method)
k = 3  # Set the number of clusters you want to find
centroids, labels = k_means(X_scaled, k)

# Plot the result (first two dimensions for visualization)
plot_clusters(X_scaled, centroids, labels, k)
