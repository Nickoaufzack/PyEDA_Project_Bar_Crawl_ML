from pathlib import Path
import pandas as pd
import openpyxl

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
df = pd.read_csv(BASE_DIR / "data" / "all_accelerometer_data_pids_13.csv")
df_a = pd.read_excel(BASE_DIR / "data" / "raw_tac"/ "CC6740 CAM Results.xlsx")

print(df_a)