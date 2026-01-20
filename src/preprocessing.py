from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

ACC_FILE = DATA_DIR / "all_accelerometer_data_pids_13.csv"
TAC_DIR = DATA_DIR / "clean_tac"

WINDOW_SECONDS = 10
SAMPLING_RATE = 40
WINDOW_SIZE = WINDOW_SECONDS * SAMPLING_RATE


def load_accelerometer_data():
    df = pd.read_csv(ACC_FILE)
    df = df.sort_values(["pid", "time"]).reset_index(drop=True)
    return df


def load_tac_data():
    tac_data = {}

    for file in TAC_DIR.glob("*_clean_TAC.csv"):
        pid = file.stem.split("_")[0]  # z.B. BK7610
        df = pd.read_csv(file)
        df = df.sort_values("timestamp").reset_index(drop=True)
        tac_data[pid] = df

    return tac_data


def create_windows_for_pid(acc_df, pid):
    df = acc_df[acc_df["pid"] == pid].copy()

    windows = []

    for start in range(0, len(df) - WINDOW_SIZE, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]

        if len(window) == WINDOW_SIZE:
            windows.append(window)

    return windows


def label_window(window, tac_df, cutoff=0.08):
    mid_time = window["time"].mean() / 1000

    index = (tac_df["timestamp"] - mid_time).abs().idxmin()
    tac_value = tac_df.loc[index, "TAC_Reading"]

    label = int(tac_value >= cutoff)

    return tac_value, label


def build_labeled_windows(acc_df, tac_data):
    X = []
    y = []
    meta = []

    for pid in tac_data.keys():
        windows = create_windows_for_pid(acc_df, pid)
        tac_df = tac_data[pid]

        for window in windows:
            tac_val, label = label_window(window, tac_df)

            X.append(window[["x", "y", "z"]].values)
            y.append(label)
            meta.append({
                "pid": pid,
                "tac": tac_val,
                "start_time": window["time"].iloc[0]
            })

    return np.array(X), np.array(y), pd.DataFrame(meta)


if __name__ == "__main__":
    acc_df = load_accelerometer_data()
    tac_data = load_tac_data()

    X, y, meta = build_labeled_windows(acc_df, tac_data)

    summary = meta.copy()
    summary["label"] = y

    summary_shuffled = summary.sample(frac=1, random_state=42).reset_index(drop=True)

    i = summary_shuffled.index[0]

    print("META INFO")
    print(summary_shuffled.loc[i])

    print("\nWINDOW SHAPE:", X[i].shape)

    window_df = pd.DataFrame(
        X[i],
        columns=["x", "y", "z"]
    )

    print("\nACCELEROMETER WINDOW (first 10 rows)")
    print(window_df.head(10))


    #raphael stinkt hööisch
