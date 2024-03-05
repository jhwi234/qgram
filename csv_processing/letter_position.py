from pathlib import Path
import pandas as pd

# Adjust the file path appropriately
file_path = Path('data/outputs/csv/brown_context_sensitive_HAPAX_split0.5_qrange6-6_prediction.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Function to find the position of the missing letter
def missing_letter_position(tested_word):
    return tested_word.find('_')

# Apply the function to find the missing letter position
data['Missing_Letter_Position'] = data['Tested_Word'].apply(missing_letter_position)

# Calculate the total length of each original word
data['Word_Length'] = data['Original_Word'].apply(len)

# Calculate the relative position of the missing letter as a percentage of the word length
data['Normalized_Missing_Letter_Position'] = data['Missing_Letter_Position'] / (data['Word_Length'] - 1)

# Use bins to group the normalized positions into intervals for easier analysis
data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=10, labels=False)

# Group by the binned normalized position and calculate mean accuracy
normalized_position_accuracy = data.groupby('Normalized_Position_Bin')['Top1_Is_Accurate'].mean().reset_index()

print(normalized_position_accuracy)
