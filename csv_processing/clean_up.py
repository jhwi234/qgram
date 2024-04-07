import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Standardize 'Tested_Word' and 'Original_Word'
        df['Tested_Word'] = df['Tested_Word'].str.lower().str.strip()
        df['Original_Word'] = df['Original_Word'].str.lower().str.strip()
        
        # Drop rows where 'Original_Word' is null
        df = df.dropna(subset=['Original_Word'])

        # Ensure boolean data types for 'In_Training_Set' and accuracy/validation flags
        boolean_columns = ['In_Training_Set'] + [f'Top{i}_Is_Valid' for i in range(1, 4)] + \
                          [f'Top{i}_Is_Accurate' for i in range(1, 4)]
        df[boolean_columns] = df[boolean_columns].astype(bool)

        # Convert 'Correct_Letter_Rank' to integer
        df['Correct_Letter_Rank'] = df['Correct_Letter_Rank'].astype(int)

        # Remove duplicates
        initial_row_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_row_count - len(df)
        
        # Validation checks (e.g., predicted letters are single characters)
        for i in range(1, 4):
            df = df[df[f'Top{i}_Predicted_Letter'].apply(lambda x: len(str(x)) == 1)]
        
        # Filter by confidence threshold (example threshold: 0.1)
        confidence_threshold = 0.1
        df = df[df['Top1_Confidence'] >= confidence_threshold]
        
        # Save the cleaned dataset, overwriting the original file
        df.to_csv(file_path, index=False)
        
        logging.info(f"Cleaned dataset overwritten at {file_path}. Duplicates removed: {duplicates_removed}.")
    except Exception as e:
        logging.error(f"Failed to clean {file_path.name} due to {e}")

data_dir = Path('data/outputs/csv')

files_to_clean = [file_path for file_path in data_dir.glob('*.csv') if 'f1' not in file_path.stem]

for file_path in files_to_clean:
    logging.info(f"Cleaning dataset: {file_path.name}")
    clean_dataset(file_path)