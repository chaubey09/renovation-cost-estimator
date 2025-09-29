import pandas as pd

try:
    df = pd.read_parquet('data/processed/processed.parquet')
    print("\nColumns in data/processed/processed.parquet:")
    print(df.columns.tolist())
except Exception as e:
    print(f"An error occurred: {e}")