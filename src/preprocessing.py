from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
df = pd.read_csv(BASE_DIR / "data" / "all_accelerometer_data_pids_13.csv")

print(df)


print("welcome")