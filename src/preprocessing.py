from pathlib import Path
import pandas as pd
import openpyxl

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
df = pd.read_csv(BASE_DIR / "data" / "all_accelerometer_data_pids_13.csv")
df_a = pd.read_csv(BASE_DIR / "data" / "clean_tac"/ "BK7610_clean_TAC.csv")

print(df_a)

print("arsch")

print("laurenz rocksass")