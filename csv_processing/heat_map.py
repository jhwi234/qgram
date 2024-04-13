import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def plot_heatmap(confusion_matrix, dataset_name, threshold=0.1, figsize=(14, 12), annot_fmt=".2f"):
    """Enhanced heatmap plotting function with customizable parameters."""
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=annot_fmt,
        cmap='viridis',  # Using 'viridis' for better color differentiation
        cbar=True,
        cbar_kws={'label': 'Frequency Proportion'}
    )
    plt.title(f'Heatmap of Most Common Substitutions for Missed Letters in {dataset_name}', fontsize=16)
    plt.xlabel('Predicted Letter', fontsize=14)
    plt.ylabel('Actual Letter', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    annotate_heatmap(ax, threshold)
    plt.show()

def annotate_heatmap(ax, threshold):
    """Apply conditional formatting to heatmap annotations based on a threshold."""
    for text in ax.texts:
        t = float(text.get_text())
        if t == 0:  # Check if the value is exactly zero
            text.set_text('')  # Clear the text
        elif t < threshold:
            text.set_color('gray')  # De-emphasize less important annotations

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