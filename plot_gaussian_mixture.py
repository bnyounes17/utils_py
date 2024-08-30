#########################################################
# This function plots contours and regions around centroids emphasizing clusters using a gaussian mixture clusterer
# inputs: - clustrer
#         - X: features
#         - resolution: integer - related to meshgrid
#         - show_ylabels: Boolean
#########################################################

import matplotlib.pyplot as plt
import numpy as np
from Voronoi_Diagram import plot_centroids # imported from the file in the same repository: Voronoi_Diagram.py

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                             np.linspace(mins[1], maxs[1], resolution))
    Z = -gm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                     norm=LogNorm(vmin=1.0, vmax=30.0),
                     levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                     norm=LogNorm(vmin=1.0, vmax=30.0),
                     levels=np.logspace(0, 2, 12),
                     linewidths=1, colors='k')
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                      linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:,0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)
    
    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)