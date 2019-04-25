"""PTP dataset generation utilities
"""
import numpy as np

"""Supported datasets

- asymmetry_classification
- toffset_regression
"""

def ds_shape(model, m):
    """Computes the shape of the dataset

    Can be used to pre-allocate the dataset matrices.

    Args:
        model : The feature model of interest
        m     : Number of examples

    """

    if (model == "asymmetry_classification"):
        return [(m, 4), (m, 1)]
    elif (model == "toffset_regression"):
        return [(m, 4), (m, 1)]
    else:
        raise ValueError("Model choice %s unknown" %(features))

def ds_features(data, model):
    """Return array of features from given data

    Args:
        data  : Dictionary with data to take features from
        model : The feature model of interest

    """

    if (model == "asymmetry_classification"):
        t4_minus_t1 = float(data["t4"] - data["t1"])
        t3_minus_t2 = float(data["t3"] - data["t2"])
        d_est       = data["d_est"]
        x_est       = data["x_est"]
        feature_vec = np.array([t4_minus_t1, t3_minus_t2, d_est, x_est])
        if (abs(data["asym"]) < 10):
            label_vec   = np.array([True])
        else:
            label_vec   = np.array([False])

        return (feature_vec, label_vec)
    elif (model == "toffset_regression"):
        t4_minus_t1 = float(data["t4"] - data["t1"])
        t3_minus_t2 = float(data["t3"] - data["t2"])
        d_est       = data["d_est"]
        x_est       = data["x_est"]
        x           = data["x"]
        feature_vec = np.array([t4_minus_t1, t3_minus_t2, d_est, x_est])
        label_vec   = np.array([x])
        return (feature_vec, label_vec)
    else:
        raise ValueError("Model choice %s unknown" %(features))


