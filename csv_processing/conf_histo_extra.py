import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_dataset(path):
    """Loads dataset from the given path, handling errors."""
    try:
        logging.info(f"Loading dataset from {path}")
        return pd.read_csv(path)
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return pd.DataFrame()  # Return an empty DataFrame if file is not found
    except Exception as e:
        logging.error(f"Error loading dataset from {path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if any other error occurs

def plot_normalized_stacked_histogram(ax, dataset, valid_color, invalid_color, label, column_name, threshold=0.50, bins=30):
    """Enhanced plotting to highlight the first bin where proportions exceed a given threshold."""
    if dataset.empty:
        ax.text(0.5, 0.5, 'Data Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
        return

    valid_predictions = dataset[dataset[column_name] == True]["Top1_Confidence"]
    invalid_predictions = dataset[dataset[column_name] == False]["Top1_Confidence"]
    bins = np.histogram(np.hstack((valid_predictions, invalid_predictions)), bins=bins)[1]
    valid_counts, _ = np.histogram(valid_predictions, bins=bins)
    invalid_counts, _ = np.histogram(invalid_predictions, bins=bins)
    
    total_counts = valid_counts + invalid_counts
    valid_proportions = valid_counts / total_counts
    invalid_proportions = invalid_counts / total_counts
    
    # Find the first bin where valid proportions exceed the threshold
    first_threshold_bin_index = np.argmax(valid_proportions >= threshold)
    first_threshold_bin = bins[first_threshold_bin_index]
    
    ax.bar(bins[:-1], valid_proportions, width=np.diff(bins), align='edge', color=valid_color, alpha=0.75, label='Valid' if column_name == "Top1_Is_Valid" else 'Accurate')
    ax.bar(bins[:-1], invalid_proportions, width=np.diff(bins), align='edge', color=invalid_color, alpha=0.65, label='Invalid' if column_name == "Top1_Is_Valid" else 'Inaccurate', bottom=valid_proportions)
    
    # Highlight the bin where valid predictions first exceed the threshold
    if valid_proportions[first_threshold_bin_index] >= threshold:
        ax.axvline(first_threshold_bin, color='black', linestyle='--')
        ax.annotate(f'{first_threshold_bin:.2f}', xy=(first_threshold_bin, 0.9), xytext=(first_threshold_bin + 0.05, 0.85),
                    arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16, color='black', fontweight='bold')
        # Add a separate legend entry for the threshold without the confidence level
        ax.plot([], [], color='black', linestyle='--', label=f'Threshold at {threshold*100:.0f}%')

    ax.set_xlabel('Top 1 Confidence', fontsize=14)
    ax.set_ylabel('Proportion', fontsize=14)
    ax.set_title(f'{label} Dataset ({column_name.split("_")[-1].capitalize()})', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()  # Add minor ticks for better readability

plt.style.use('seaborn-v0_8-colorblind')

# Define dataset paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Load datasets
loaded_datasets = {name: load_dataset(Path(filepath)) for name, filepath in datasets.items()}
n_datasets = len(loaded_datasets)
n_cols = 2
n_rows = (n_datasets + n_cols - 1) // n_cols  # Ensure enough rows

# Function to plot and save figures for given column_name
def plot_and_save_figures(column_name, filename):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), squeeze=False)
    colors = [plt.get_cmap('tab10')(i) for i in range(n_datasets)]

    combined_data = []

    for (label, dataset), color, ax in zip(loaded_datasets.items(), colors, axs.flatten()):
        plot_normalized_stacked_histogram(ax, dataset, color, 'tab:gray', label, column_name)
        if not dataset.empty:
            combined_data.append(dataset)

    for ax in axs.flatten()[len(loaded_datasets):]:
        ax.set_visible(False)

    plt.tight_layout(pad=2.0)
    output_dir = Path('output/confs')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{filename}.png')
    plt.close(fig)

    # Combine all datasets and plot the combined histogram
    if combined_data:
        combined_dataset = pd.concat(combined_data, ignore_index=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_normalized_stacked_histogram(ax, combined_dataset, 'tab:blue', 'tab:gray', 'Combined', column_name)
        fig.savefig(output_dir / f'combined_{filename}.png')
        plt.close(fig)

# Create plots for "Top1_Is_Accurate"
plot_and_save_figures("Top1_Is_Accurate", 'normalized_accurate_stacked_histograms_extra')

# Create plots for "Top1_Is_Valid"
plot_and_save_figures("Top1_Is_Valid", 'normalized_valid_stacked_histograms_extra')
