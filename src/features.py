import librosa
import numpy as np
from scipy.stats import skew, kurtosis


def extract_features(window):
    """
    extract features from one accelerometer window
    """
    features = []

    x_vals = window["x"].values
    y_vals = window["y"].values
    z_vals = window["z"].values

    for axis in ["x", "y", "z"]:
        data = window[axis].values

        # Mean
        features.append(np.mean(data))

        # Variance
        features.append(np.std(data))

        # Min/Max Raw
        features.append(np.min(data))
        features.append(np.max(data))

        # Root-mean-square
        features.append(np.sqrt(np.mean(data ** 2)))

        # Median
        features.append(np.median(data))

        # Zero Crossing Rate
        signs = np.sign(data)
        zcr = np.mean(signs[:-1] != signs[1:])
        features.append(zcr)

        # Min/Max Absolute
        abs_data = np.abs(data)
        features.append(np.min(abs_data))
        features.append(np.max(abs_data))

    # Magnitude 
    mag = np.sqrt(x_vals ** 2 + y_vals ** 2 + z_vals ** 2)
    features.append(np.mean(mag))
    features.append(np.std(mag))

    # Cross-axis
    c_xy = np.corrcoef(x_vals, y_vals)[0, 1]
    c_xz = np.corrcoef(x_vals, z_vals)[0, 1]
    c_yz = np.corrcoef(y_vals, z_vals)[0, 1]

    features.extend(np.nan_to_num([c_xy, c_xz, c_yz]))

    return np.array(features)


def extract_features_for_all_windows(X):
    """
    extract features for all windows
    """
    features = []
    for window in X:
        features.append(extract_features(window))
    return np.array(features)
