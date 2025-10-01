import os
import pandas as pd
import numpy as np

# Set the directory containing your CSV files
csv_folder = "history_manual_data"

# Loop through all files in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(csv_folder, filename)
        try:
            df = pd.read_csv(filepath)
            if df.isnull().values.any():
                nan_rows, nan_cols = np.where(df.isnull().values)
                print(f"⚠️ NaNs found in '{filename}'")
                print(f"   → Rows: {np.unique(nan_rows)}")
                print(f"   → Columns: {df.columns[np.unique(nan_cols)].tolist()}")
            else:
                print(f"✅ No NaNs in '{filename}'")
        except Exception as e:
            print(f"❌ Error reading '{filename}': {e}")
