import numpy as np


def extract_basic_features(window):
    """
    extract mean, std, min, max and RMS from one accelerometer window
    """
    features = []

    for axis in ["x", "y", "z"]:
        data = window[axis].values

        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.min(data))
        features.append(np.max(data))
        features.append(np.sqrt(np.mean(data ** 2)))  # RMS

    return np.array(features)


# ToDo: do that shit
def extract_time_features(window):
    return np.array([])


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
