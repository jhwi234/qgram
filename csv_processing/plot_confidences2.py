import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

# Setting a color-blind friendly style
plt.style.use('seaborn-v0_8-colorblind')

# Dictionary mapping dataset names to their CSV file paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv'
}

# Load the datasets into a dictionary for easy access
loaded_datasets = {name: pd.read_csv(Path(filepath)) for name, filepath in datasets.items()}

# Setup the subplot grid
n_datasets = len(loaded_datasets)
n_cols = 3
n_rows = np.ceil(n_datasets / n_cols).astype(int)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

def plot_correct_predictions_histogram(ax, dataset, color, label):
    """
    Plots histograms for both correct and total predictions on the given axes.
    Highlights the confidence levels for correct predictions.
    """
    # Filter the dataset for correct and incorrect predictions based on 'Top1_Is_Accurate' column
    correct_confidences = dataset[dataset["Top1_Is_Accurate"] == 1]["Top1_Confidence"]
    incorrect_confidences = dataset[dataset["Top1_Is_Accurate"] == 0]["Top1_Confidence"]
    
    # Plot histogram for all Top 1 Confidence values
    ax.hist(dataset["Top1_Confidence"], bins=30, color=color, alpha=0.5, label=f'All Predictions in {label}')
    # Overlay with histogram for correct predictions
    ax.hist(correct_confidences, bins=30, color='yellow', alpha=0.7, label=f'Correct Predictions in {label}')
    
    ax.set(title=f'{label} Dataset', xlabel='Top 1 Confidence', ylabel='Frequency')
    ax.legend()

# Plot histograms with overlays for each dataset
for i, ((label, dataset), ax) in enumerate(zip(loaded_datasets.items(), axs.flatten())):
    plot_correct_predictions_histogram(ax, dataset, colors[i % len(colors)], label)

# Remove unused subplots if the total number of datasets doesn't fill the grid
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.show()