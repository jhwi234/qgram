import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        
        df['Tested_Word'] = df['Tested_Word'].str.lower().str.strip()
        df['Original_Word'] = df['Original_Word'].str.lower().str.strip()
        
        # Drop rows where 'Original_Word' is null
        df.dropna(subset=['Original_Word'], inplace=True)

        # Ensure boolean data types
        df['In_Training_Set'] = df['In_Training_Set'].astype(bool)
        for i in range(1, 4):
            df[f'Top{i}_Is_Valid'] = df[f'Top{i}_Is_Valid'].astype(bool)
            df[f'Top{i}_Is_Accurate'] = df[f'Top{i}_Is_Accurate'].astype(bool)

        # Ensure 'Correct_Letter_Rank' is integer
        df['Correct_Letter_Rank'] = df['Correct_Letter_Rank'].astype(int)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Save the cleaned dataset
        cleaned_file_path = file_path.parent / f"{file_path.stem}{file_path.suffix}"
        df.to_csv(cleaned_file_path, index=False)
        logging.info(f"Cleaned dataset saved to {cleaned_file_path}")
    except Exception as e:
        logging.error(f"Failed to clean {file_path.name} due to {e}")

data_dir = Path('data/outputs/csv')

files_to_clean = [file_path for file_path in data_dir.glob('*.csv') if 'f1' not in file_path.stem]

for file_path in files_to_clean:
    logging.info(f"Cleaning dataset: {file_path.name}")
    clean_dataset(file_path)