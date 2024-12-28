import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Generate sample data
np.random.seed(42)
data = np.random.rand(10, 2)  # 10 samples with 2 features

# Perform hierarchical clustering
linked = linkage(data, method='ward')  # Using Ward's method

# Create a dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=np.arange(1, 11), distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Form flat clusters
threshold = 1.5  # Set the distance threshold
clusters = fcluster(linked, threshold, criterion='distance')

# Print the cluster assignments
print("Cluster assignments:", clusters)
