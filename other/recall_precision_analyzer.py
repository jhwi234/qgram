import pandas as pd
from pathlib import Path

# Define the directory where the CSV files are located
csv_dir = Path('data/outputs/csv/')

# Check if the directory exists
if not csv_dir.exists():
    print(f"Directory not found: {csv_dir}")
    exit()

print(f"Processing files in: {csv_dir}")

def average_precision_recall_per_file(csv_directory):
    results = {}

    for file in csv_directory.glob("*recall*.csv"):
        print(f"Processing file: {file.name}")
        df = pd.read_csv(file)

        # Strip whitespace from column names and character column
        df.columns = df.columns.str.strip()
        df['Character'] = df['Character'].str.strip()

        # Check required columns are present and DataFrame is not empty
        if not df.empty and {'Character', 'Recall', 'Precision'}.issubset(df.columns):
            df['Average Combined'] = (df['Recall'] + df['Precision']) / 2
            sorted_df = df.sort_values(by='Average Combined', ascending=False)
            results[file.name] = sorted_df[['Character', 'Recall', 'Precision', 'Average Combined']]

        else:
            print(f"Skipped {file.name} due to missing columns or empty DataFrame")

    return results

results_per_file = average_precision_recall_per_file(csv_dir)

for file, df in results_per_file.items():
    print(f"\nResults for {file}:")
    print(f"{'Character':>10} | {'Recall':>6} | {'Precision':>9} | {'Average Combined':>15}")
    print("-" * 55)
    print(df.to_string(index=False, formatters={
        'Recall': '{:0.4f}'.format,
        'Precision': '{:0.4f}'.format,
        'Average Combined': '{:0.4f}'.format
    }))
