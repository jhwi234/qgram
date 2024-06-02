import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set(style="whitegrid", context="notebook", palette="Paired")
plt.style.use('seaborn-v0_8-colorblind')

# Define dataset paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

def ensure_directory_exists(directory):
    """
    Ensure that the given directory exists, create it if it doesn't.
    """
    os.makedirs(directory, exist_ok=True)

def plot_confidence_intervals(data, title, figure_size=(12, 8), font_size=12, ci=95, save_path=None):
    """
    Plots confidence intervals for the given data using a bar chart.
    """
    colors = get_color_palette()
    fig, ax = plt.subplots(figsize=figure_size)
    metrics = ['Top1', 'Top2', 'Top3']
    accuracies = ['Accurate', 'Inaccurate']
    bar_width = 0.35
    index = np.arange(len(metrics) * 2)

    for i, metric in enumerate(metrics):
        for j, accuracy in enumerate(accuracies):
            series = data[data[f'{metric}_Is_Accurate'] == (accuracy == 'Accurate')][f'{metric}_Confidence']
            if series.isnull().all():
                logger.warning(f"No data available for {accuracy} {metric}. Skipping.")
                continue
            lower, upper, mean_confidence = bootstrap_confidence(series, ci=ci)
            ax.bar(index[i * 2 + j], mean_confidence, bar_width, color=colors[i][j], label=f'{accuracy} {metric}')
            ax.text(index[i * 2 + j], mean_confidence, f'{mean_confidence:.2%}', ha='center', va='bottom', fontsize=font_size)

    ax.set_xlabel('Metrics', fontsize=font_size)
    ax.set_ylabel('Mean Confidence', fontsize=font_size)
    ax.set_title(f'Confidence Intervals for {title}', fontsize=font_size)
    ax.set_xticks(index)
    ax.set_xticklabels([f'{acc} {met}' for met in metrics for acc in accuracies], rotation=45, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    plt.tight_layout()
    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))  # Ensure directory exists
        plt.savefig(save_path)  # Save the plot as PNG
    plt.show()

def get_color_palette():
    base_colors = sns.color_palette("Paired", 12)
    return [[base_colors[i + 1], base_colors[i]] for i in range(0, len(base_colors), 2)]

def bootstrap_confidence(data, n_bootstrap=1000, ci=95):
    if data.isnull().all():
        return None, None, None

    bootstrap_samples = np.random.choice(data, (len(data), n_bootstrap), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=0)
    confidence_bounds = np.percentile(bootstrap_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    mean_confidence = np.mean(data)
    return confidence_bounds[0], confidence_bounds[1], mean_confidence

def load_and_preprocess_data(path):
    try:
        data = pd.read_csv(Path(path))
        if data.empty:
            logger.warning(f"No data to process after loading from {path}.")
            return None
        data[['Top1_Is_Accurate', 'Top2_Is_Accurate', 'Top3_Is_Accurate']] = data[
            ['Top1_Is_Accurate', 'Top2_Is_Accurate', 'Top3_Is_Accurate']
        ].astype(bool)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        return None

def process_datasets(datasets):
    all_data = []
    for name, path in datasets.items():
        logger.info(f"Processing dataset: {name}")
        if not Path(path).exists():
            logger.warning(f"File not found: {path}. Skipping dataset: {name}")
            continue
        data_preprocessed = load_and_preprocess_data(path)
        if data_preprocessed is not None:
            all_data.append(data_preprocessed)
            save_path = f'output/bootstrap/{name}.png'  # Define the save path for the PNG
            plot_confidence_intervals(data_preprocessed, name, save_path=save_path)  # Pass the save path
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        save_path = 'output/bootstrap/All_Datasets.png'
        plot_confidence_intervals(combined_data, 'All Datasets', save_path=save_path)

def main():
    process_datasets(datasets)

if __name__ == "__main__":
    main()
