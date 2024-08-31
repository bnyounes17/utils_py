#########################################################
# This function detects anomalies or outliers from a dataset
# using a model like Gaussian Mixture and by defining densities and
# density threshold
#
# If you notice that you get too many false positives (i.e., perfectly good products
# that are flagged as defective), you can lower the threshold. Conversely, if you have too
# many false negatives (i.e., defective products that the system does not flag as defective)
# you can increase the threshold. 
#
# inputs: - model
#         - X: features
#         - percent: percentage of threshold
#
# output: anomalies (an array of instances)
#########################################################

import numpy as np

# Identify anomalies from a density threshold
def anomalies_detection(model, X, percent=1):
    densities = model.score_samples(X)
    density_threshold = np.percentile(densities, percent)
    anomalies = X[densities < density_threshold]
    return anomalies