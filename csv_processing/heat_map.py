import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set a color-blind friendly style for plots
plt.style.use('seaborn-v0_8-colorblind')

def load_dataset(filepath):
    """Load dataset from a specified filepath."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def filter_mispredictions(data):
    """Filter dataset to only include mispredictions."""
    if data is not None:
        return data[data['Correct_Letter'] != data['Top1_Predicted_Letter']]
    return pd.DataFrame()

def calculate_confusion_matrix(mispredictions):
    """Calculate and normalize the confusion matrix for mispredictions."""
    confusion_matrix = pd.crosstab(mispredictions['Correct_Letter'], mispredictions['Top1_Predicted_Letter'])
    # Set diagonal entries to NaN
    for letter in confusion_matrix.index:
        if letter in confusion_matrix.columns:
            confusion_matrix.at[letter, letter] = None
    return confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

def plot_heatmap(confusion_matrix, dataset_name):
    """Plot a heatmap for the confusion matrix with enhanced visual features."""
    plt.figure(figsize=(12, 10))  # Larger figure to improve readability
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,  # Annotate each cell with the numeric value
        fmt=".2f",   # Formatting annotations to 2 decimal places
        cbar=True,
        cbar_kws={'label': 'Frequency Proportion'}  # Labeling the color bar
    )
    plt.title(f'Heatmap of Most Common Substitutions for Missed Letters in {dataset_name}', fontsize=14)
    plt.xlabel('Predicted Letter', fontsize=12)
    plt.ylabel('Actual Letter', fontsize=12)
    plt.xticks(rotation=45)  # Rotating x-ticks to avoid text overlapping
    plt.yticks(rotation=0)   # Ensure y-ticks are readable
    plt.tight_layout()  # Adjust layout to make room for tick labels

    # Apply threshold for annotations to enhance visibility of significant values
    threshold = 0.1  # Define the threshold for significant values
    for text in ax.texts:
        t = float(text.get_text())
        if t < threshold:  # Clear annotation if value is below threshold
            text.set_text('')
    
    plt.show()

# Paths to datasets
dataset_paths = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Process each dataset
for name, path in dataset_paths.items():
    data = load_dataset(Path(path))
    mispredictions = filter_mispredictions(data)
    if not mispredictions.empty:
        confusion_matrix = calculate_confusion_matrix(mispredictions)
        plot_heatmap(confusion_matrix, name)