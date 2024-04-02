import pandas as pd
from pathlib import Path

# Define dataset paths
datasets = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
}

def clean_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Ensure consistent casing
    df['Tested_Word'] = df['Tested_Word'].str.lower()
    df['Original_Word'] = df['Original_Word'].str.lower()
    
    # Trim whitespace
    df['Tested_Word'] = df['Tested_Word'].str.strip()
    df['Original_Word'] = df['Original_Word'].str.strip()
    
    # Drop rows where 'Original_Word' is null
    df.dropna(subset=['Original_Word'], inplace=True)
    
    # Validate data types
    # Ensuring 'In_Training_Set' is treated as a boolean
    df['In_Training_Set'] = df['In_Training_Set'].astype(bool)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Save the cleaned dataset back to disk
    cleaned_file_path = file_path.parent / f"{file_path.stem}_cleaned{file_path.suffix}"
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_file_path}")

# Iterate over each dataset, clean it, and save the cleaned version
for name, path in datasets.items():
    print(f"Cleaning dataset: {name}")
    clean_dataset(path)
