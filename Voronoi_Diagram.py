#############################################################
# Voronoi Diagram: Clusters' highlight
# Voronoi tessellation: Cluster’s decision boundaries where each centroid is represented with an (x) inside a circle)
# The functions below provide a representation of clusters such that each cluster'region is specified with a specific color and is delimited with boundaries
# Each cluster's centroid is represented too
#############################################################


import matplotlib.pyplot as plt
import numpy as np

# Plot the clusters
# If many features, it is better to apply a PCA to detect the most 2 or 3 important features
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

# Represent the centtroids with a (x) inside a (o)
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    
    # centroid represented with a cross inside a circle
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=35, linewidths=8,
               color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=2, linewidths=12,
               color=cross_color, zorder=11, alpha=1)

# Define boundaries for each cluster
# The input clusterer can be a model after applying KMeans algorithm for instance
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True, show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                        np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Define colorful regions per cluster
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    # Define line boundaries
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
        
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)





