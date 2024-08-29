##########################################################################
# This function calculates the Probability Density Function (PDF)
# applied to a certain model
# inputs: - model: (ex. Gaussian Mixture)
#         - resolution: sets the granularity of the grid
#         - grid_param: value to form a grid of coordinates
# output: a probability array
##########################################################################


import numpy as np

def pfd_probas(model, resolution=100, grid_param=10):
    
    grid = np.arange(-grid_param, grid_param, 1 / resolution)
    xx, yy = np.meshgrid(grid, grid)
    X_full = np.vstack([xx.ravel(), yy.ravel()]).T

    pdf = np.exp(model.score_samples(X_full))
    pdf_probas = pdf * (1/resolution)**2

    return pdf_probas