import pandas as pd
import os

# Define the directory where the CSV files are located
csv_dir = 'data/outputs/csv/'

# Check if the directory exists
if not os.path.exists(csv_dir):
    print(f"Directory not found: {csv_dir}")
    exit()

print(f"Processing files in: {csv_dir}")

# Function to compute average of precision and recall from CSV files with 'recall' in their name
def average_precision_recall_per_file(csv_directory):
    results = {}

    for file in os.listdir(csv_directory):
        if 'recall' in file and file.endswith('.csv'):
            print(f"Processing file: {file}")
            file_path = os.path.join(csv_directory, file)
            df = pd.read_csv(file_path)

            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            combined_metrics = []

            if not df.empty and 'Character' in df.columns and 'Recall' in df.columns and 'Precision' in df.columns:
                for _, row in df.iterrows():
                    char = row['Character'].strip()
                    recall = row['Recall']
                    precision = row['Precision']
                    average_combined = (recall + precision) / 2  # Compute average
                    combined_metrics.append((char, recall, precision, average_combined))

                # Sort by combined average score in descending order
                combined_metrics.sort(key=lambda x: x[3], reverse=True)

                # Save sorted results for this file
                results[file] = combined_metrics

            else:
                print(f"Skipped {file} due to missing columns or empty DataFrame")

    return results

# Calling the function
results_per_file = average_precision_recall_per_file(csv_dir)

# Printing the results in a readable format
for file, metrics in results_per_file.items():
    print(f"\nResults for {file}:")
    print(f"{'Character':>10} | {'Recall':>6} | {'Precision':>9} | {'Average Combined':>15}")
    print("-" * 55)
    for char, recall, precision, average_combined in metrics:
        print(f"{char:>10} | {recall:6.4f} | {precision:9.4f} | {average_combined:15.4f}")