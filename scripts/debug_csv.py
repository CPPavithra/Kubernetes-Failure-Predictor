import pandas as pd
import os

CSV_PATH = "../data/merged_data.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"The file {CSV_PATH} does not exist. Check the path!")
print("File found! Proceeding with CSV loading...\n")

try:
    df = pd.read_csv(CSV_PATH)
    print("CSV loaded\n")
except Exception as e:
    raise RuntimeError(f"Failed: {e}")

print("Columns from CSV:", df.columns.tolist())
print("\nSample Data:\n", df.head())

df.columns = df.columns.str.strip()  #removing space
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  #replace the space with _
df.columns = df.columns.str.lower()  #converting to lower case

print("\n Cleaned Columns:", df.columns.tolist())
column_name = "container_restart_count"

if column_name not in df.columns:
    raise ValueError(f"Column '{column_name}' not found.Available: {df.columns.tolist()}")
else:
    print(f"\n Column '{column_name}' found successfully")

try:
    df_fixed = pd.read_csv(CSV_PATH, encoding="utf-8", delimiter=",", engine="python")
    print("\nCSV reloaded.")
except Exception as e:
    print(f"⚠️Could not reload: {e}")

# Step 7: Manually print first few rows as raw text
print("\nPreview:")
with open(CSV_PATH, "r", encoding="utf-8") as f:
    for _ in range(5):  # Print first 5 lines
        print(f.readline().strip())

