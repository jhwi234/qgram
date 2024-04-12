import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Dictionary mapping dataset names to their CSV file paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Load the datasets into a dictionary for easy access
loaded_datasets = {name: pd.read_csv(Path(filepath)) for name, filepath in datasets.items()}

# Setting a color-blind friendly style
plt.style.use('seaborn-v0_8-colorblind')

# Setup the subplot grid with numpy
n_datasets = len(loaded_datasets)
n_cols = 3
n_rows = np.ceil(n_datasets / n_cols).astype(int)  # Calculate the number of rows needed

fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)  # Create a grid of subplots
colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for distinct colors

def plot_valid_predictions_histogram(ax, dataset, base_color, label):
    """
    Improved plotting function for valid and invalid predictions histograms
    with normalized and stacked histograms, displaying raw counts in the legend.
    """
    valid_predictions = dataset[dataset["Top1_Is_Valid"] == True]["Top1_Confidence"]
    invalid_predictions = dataset[dataset["Top1_Is_Valid"] == False]["Top1_Confidence"]
    
    # Unified bin edges for direct comparison
    bins = np.histogram(np.hstack((valid_predictions, invalid_predictions)), bins=30)[1]
    
    # Calculate normalized counts
    valid_counts, _ = np.histogram(valid_predictions, bins=bins)
    invalid_counts, _ = np.histogram(invalid_predictions, bins=bins)
    
    # Normalize the counts
    valid_heights = valid_counts / valid_counts.sum()
    invalid_heights = invalid_counts / invalid_counts.sum()
    
    # Stack histograms
    ax.bar(bins[:-1], valid_heights, width=np.diff(bins), align='edge', color=base_color, alpha=0.75, label=f'Valid (Count: {valid_counts.sum()})', edgecolor='black')
    ax.bar(bins[:-1], invalid_heights, width=np.diff(bins), align='edge', color='gray', alpha=0.65, label=f'Invalid (Count: {invalid_counts.sum()})', edgecolor='black', bottom=valid_heights)
    
    ax.set_xlabel('Top 1 Confidence', fontsize=14)
    ax.set_ylabel('Normalized Frequency', fontsize=14)
    ax.set_title(f'{label} Dataset', fontsize=14)
    ax.legend(fontsize=12)

# Apply the improved plotting function to each dataset
for i, ((label, dataset), ax) in enumerate(zip(loaded_datasets.items(), axs.flatten())):
    plot_valid_predictions_histogram(ax, dataset, colors[i % len(colors)], label)

# Hide unused subplot axes for a cleaner look
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axs.flatten()[j])

plt.tight_layout()  # Adjust layout for a neat presentation
plt.show()
