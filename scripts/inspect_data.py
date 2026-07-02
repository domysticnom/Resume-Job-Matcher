from pathlib import Path
import pandas as pd

raw_dir = Path("data/raw")

csv_files = list(raw_dir.glob("*.csv"))

print("CSV files found:")
for file in csv_files:
    print("-", file)

print("\nInspecting files...\n")

for file in csv_files:
    print("=" * 80)
    print(f"File: {file}")
    
    df = pd.read_csv(file)
    
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3))
    print()