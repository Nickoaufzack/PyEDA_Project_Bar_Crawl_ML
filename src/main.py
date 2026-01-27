import time
import numpy as np
from pathlib import Path

from src.preprocessing import load_accelerometer_data, load_tac_data, build_labeled_windows, RESULTS_DIR
from src.features import extract_all_features_for_all_windows
from src.model import train_random_forest
from src.evaluate import evaluate_model

if __name__ == "__main__":
    start_time = time.time()

    print("Loading data...")
    acc_df = load_accelerometer_data()
    tac_data = load_tac_data()

    print("\nBuilding labeled windows...")
    X, y, meta = build_labeled_windows(acc_df, tac_data)

    print("\nExtracting features...")
    X_features = extract_all_features_for_all_windows(X)

    print("Feature matrix shape:", X_features.shape)
    print("Label distribution:", np.bincount(y))


    print("\nTraining Random Forest...")
    model, X_test, y_test, y_pred = train_random_forest(
        X_features,
        y,
        meta,
        test_size=0.25,
        random_state=69
    )

    print("\nPrediction sanity check:")
    print("y_test shape:", y_test.shape)
    print("y_pred shape:", y_pred.shape)
    print("First 20 predictions:", y_pred[:20])

    print("Saving Report...")
    REPORT_PATH = RESULTS_DIR / "evaluation_report.pdf"

    evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        output_path=REPORT_PATH
    )

    print(f"\nEvaluation report saved to {REPORT_PATH}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")