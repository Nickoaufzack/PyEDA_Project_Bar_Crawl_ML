import numpy as np

from src.features import extract_all_features, extract_all_features_for_all_windows
from src.preprocessing import load_accelerometer_data, load_tac_data, build_labeled_windows

if __name__ == "__main__":
    acc_df = load_accelerometer_data()
    tac_data = load_tac_data()

    X, y, meta = build_labeled_windows(acc_df, tac_data)

    X_features = extract_all_features_for_all_windows(X)
    print(X_features.shape)