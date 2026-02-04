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
    """
    slice accelerometer data into 10-second windows for a participant
    """
    df = acc_df[acc_df["pid"] == pid].copy()

    windows = []

    for start in range(0, len(df) - WINDOW_SIZE, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]

        if len(window) == WINDOW_SIZE:
            windows.append(window)

    return windows


def label_window(window, tac_df, cutoff=0.08):
    """
    assign label sober (0) or intoxicated (1) based on TAC value
    """
    mid_time = window["time"].mean() / 1000

    index = (tac_df["timestamp"] - mid_time).abs().idxmin()
    tac_value = tac_df.loc[index, "TAC_Reading"]

    label = int(tac_value >= cutoff)

    return tac_value, label


def build_labeled_windows(acc_df, tac_data):
    """
    build labeled windows for all participants
    """
    X = []
    y = []
    meta = []

    for pid in tac_data.keys():
        windows = create_windows_for_pid(acc_df, pid)
        tac_df = tac_data[pid]

        for window in windows:
            tac_val, label = label_window(window, tac_df)

            X.append(window[["x", "y", "z"]])
            y.append(label)
            meta.append({
                "pid": pid,
                "tac": tac_val,
                "start_time": window["time"].iloc[0]
            })

    return X, np.array(y), pd.DataFrame(meta)
