import pandas as pd
from pathlib import Path

# Suppress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define dataset paths
datasets = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
}

# Load and concatenate all datasets
combined_data = pd.concat([pd.read_csv(file) for file in datasets.values()], ignore_index=True)

# Calculate the relative position of the missing letter
combined_data['Missing_Letter_Position'] = combined_data['Tested_Word'].apply(lambda x: x.find('_'))
combined_data['Word_Length'] = combined_data['Tested_Word'].apply(lambda x: len(x) - 1)
combined_data['Relative_Position'] = combined_data['Missing_Letter_Position'] / combined_data['Word_Length']

# Define word length categories and categorize words
bins = [0, 4, 8, float('inf')]
labels = ['Short', 'Medium', 'Long']
combined_data['Word_Length_Category'] = pd.cut(combined_data['Word_Length'], bins=bins, labels=labels, right=False)

# Calculate the average accuracy for each relative position range, controlled by word length
# Include observed=True to address the FutureWarning
accuracy_by_length_and_position = combined_data.groupby(
    ['Word_Length_Category', pd.cut(combined_data['Relative_Position'], bins=10)],
    observed=True  # This addresses the FutureWarning by explicitly stating behavior for categorical data
)['Top1_Is_Accurate'].mean().unstack(level=0)

# Print the result clearly
print("Average prediction accuracy across different relative positions for Short, Medium, and Long words:")
print(accuracy_by_length_and_position)
