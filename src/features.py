import numpy as np
from scipy.stats import skew, kurtosis


def extract_basic_features(window):
    """
    extract mean, std, min, max and RMS from one accelerometer window
    """
    features = []

    for axis in ["x", "y", "z"]:
        data = window[axis].values

        features.append(np.mean(data))
        features.append(np.std(data))
        # Min/Max Raw
        features.append(np.min(data))
        features.append(np.max(data))
        # Root-mean-square (RMS)
        features.append(np.sqrt(np.mean(data ** 2)))

    return np.array(features)


def extract_time_features(window):
    features = []

    for axis in ["x", "y", "z"]:
        data = window[axis].values

        features.append(np.median(data))

        # Zero Crossing Rate
        signs = np.sign(data)
        zcr = np.mean(signs[:-1] != signs[1:])
        features.append(zcr)

        # Min/Max Absolute
        abs_data = np.abs(data)
        features.append(np.min(abs_data))
        features.append(np.max(abs_data))

        features.append(skew(data))
        features.append(kurtosis(data))

    return np.array(features)


# ToDo: do the shit
def extract_frequency_features(window):
    return np.array([])


def extract_all_features(window):
    """
    Extract all features for one window
    """
    return np.concatenate([extract_basic_features(window),
                           extract_time_features(window),
                           extract_frequency_features(window)])


def extract_all_features_for_all_windows(X):
    """
    extract all features for all windows
    """
    features = []
    for window in X:
        features.append(extract_all_features(window))
    return np.array(features)
