import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def load_dataset(path):
    """ Loads dataset from the given path, handling errors. """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return pd.DataFrame()  # Return an empty DataFrame if file is not found

def plot_normalized_stacked_histogram(ax, dataset, base_color, label, validity_threshold=0.90):
    """Enhanced plotting to highlight the first bin where valid predictions exceed the invalid ones by a given threshold."""
    if dataset.empty:
        ax.text(0.5, 0.5, 'Data Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return
    
    valid_predictions = dataset[dataset["Top1_Is_Valid"] == True]["Top1_Confidence"]
    invalid_predictions = dataset[dataset["Top1_Is_Valid"] == False]["Top1_Confidence"]
    bins = np.histogram(np.hstack((valid_predictions, invalid_predictions)), bins=30)[1]  # Determine bins for histogram
    valid_counts, _ = np.histogram(valid_predictions, bins=bins)
    invalid_counts, _ = np.histogram(invalid_predictions, bins=bins)
    
    total_counts = valid_counts + invalid_counts
    valid_proportions = valid_counts / total_counts
    invalid_proportions = invalid_counts / total_counts
    
    # Find the first bin where valid proportions exceed the set threshold
    first_threshold_bin_index = np.argmax(valid_proportions >= validity_threshold)
    first_threshold_bin = bins[first_threshold_bin_index]
    
    ax.bar(bins[:-1], valid_proportions, width=np.diff(bins), align='edge', color=base_color, alpha=0.75, label='Valid')
    ax.bar(bins[:-1], invalid_proportions, width=np.diff(bins), align='edge', color='gray', alpha=0.65, label='Invalid', bottom=valid_proportions)
    
    # Highlight the bin where valid predictions first exceed the threshold
    if valid_proportions[first_threshold_bin_index] >= validity_threshold:
        ax.axvline(first_threshold_bin, color='black', linestyle='--', label=f'Threshold at {validity_threshold*100:.0f}%: {first_threshold_bin:.2f}')

    ax.set_xlabel('Top 1 Confidence')
    ax.set_ylabel('Proportion')
    ax.set_title(f'{label} Dataset')
    ax.legend()

    return ax  # Returning ax for further manipulation if needed

plt.style.use('seaborn-v0_8-colorblind')

datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

loaded_datasets = {name: load_dataset(Path(filepath)) for name, filepath in datasets.items()}
n_datasets = len(loaded_datasets)
n_cols = 2
n_rows = (n_datasets + n_cols - 1) // n_cols  # Ensure enough rows
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)

colors = [plt.get_cmap('tab10')(i) for i in range(n_datasets)]

for (label, dataset), color, ax in zip(loaded_datasets.items(), colors, axs.flatten()):
    plot_normalized_stacked_histogram(ax, dataset, color, label)

for ax in axs.flatten()[len(loaded_datasets):]:
    ax.set_visible(False)  # Hide unused axes

plt.tight_layout()

# Define the output directory and ensure it exists
output_dir = Path('output/confs')
output_dir.mkdir(parents=True, exist_ok=True)

# Save the figure
fig.savefig(output_dir / 'normalized_valid_stacked_histograms_extra.png')
plt.close(fig)  # Close the plot to free up memory
