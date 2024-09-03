###############################################
# Below are activation functions
# We list here 3 of them:
# - Sigmoid
# - ReLU
# - Derivative
###############################################

import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps)) / (2 * eps)