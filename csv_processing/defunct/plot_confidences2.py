import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

# Setting a color-blind friendly style
plt.style.use('seaborn-v0_8-colorblind')

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

# Setup the subplot grid
n_datasets = len(loaded_datasets)
n_cols = 3
n_rows = np.ceil(n_datasets / n_cols).astype(int)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)  # Create a grid of subplots
colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for distinct colors

def plot_accurate_vs_inaccurate_predictions_histogram(ax, dataset, base_color, label):
    """
    Plots normalized histograms for accurate vs. inaccurate predictions on the given axes,
    includes raw counts in the legend.
    """
    accurate_confidences = dataset[dataset["Top1_Is_Accurate"] == True]["Top1_Confidence"]
    inaccurate_confidences = dataset[dataset["Top1_Is_Accurate"] == False]["Top1_Confidence"]
    
    # Unified bin edges for direct comparison
    bins = np.histogram(np.hstack((accurate_confidences, inaccurate_confidences)), bins=30)[1]
    
    # Calculate normalized counts
    accurate_counts, _ = np.histogram(accurate_confidences, bins=bins)
    inaccurate_counts, _ = np.histogram(inaccurate_confidences, bins=bins)
    
    # Normalize the counts
    accurate_heights = accurate_counts / accurate_counts.sum()
    inaccurate_heights = inaccurate_counts / inaccurate_counts.sum()
    
    # Stack histograms
    ax.bar(bins[:-1], accurate_heights, width=np.diff(bins), align='edge', color=base_color, alpha=0.75, label=f'Accurate (Count: {accurate_counts.sum()})', edgecolor='black')
    ax.bar(bins[:-1], inaccurate_heights, width=np.diff(bins), align='edge', color='gray', alpha=0.65, label=f'Inaccurate (Count: {inaccurate_counts.sum()})', edgecolor='black', bottom=accurate_heights)
    
    ax.set_xlabel('Top 1 Confidence', fontsize=14)
    ax.set_ylabel('Normalized Frequency', fontsize=14)
    ax.set_title(f'{label} Dataset', fontsize=14)
    ax.legend(fontsize=12)

# Plot histograms with overlays for each dataset
for i, ((label, dataset), ax) in enumerate(zip(loaded_datasets.items(), axs.flatten())):
    plot_accurate_vs_inaccurate_predictions_histogram(ax, dataset, colors[i % len(colors)], label)

# Remove unused subplots if the total number of datasets doesn't fill the grid
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.show()
