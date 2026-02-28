import pandas as pd

# Load the CSV file
from pathlib import Path

# Project root = directory where main.py lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Input file
file_path = PROJECT_ROOT / "data" / "open_meteo_51.78N10.35E563m.csv"
df = pd.read_csv(file_path)

# Total number of rows
total_rows = len(df)

# Divide by 4
n = total_rows // 4

print(f"Total rows: {total_rows}")
print(f"n (total_rows // 4): {n}\n")

# Dictionary to store selected rows for each column
selected_rows = {}

# Process each column
for column in df.columns:
    # Sort the entire DataFrame by this column (descending)
    sorted_df = df.sort_values(by=column, ascending=False)

    # Select top n rows (ALL columns preserved)
    top_n_rows = sorted_df.head(n)

    # Save to dictionary
    selected_rows[column] = top_n_rows

    # Print result
    print(f"Sorted by column: {column}")
    print(top_n_rows)
    print("=" * 60)