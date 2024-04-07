import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Applying improved style settings within function to avoid global side-effects
def set_seaborn_style():
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("Paired")

# Define the datasets paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv'
}

def preprocess_data(data):
    """Convert accuracy indicators to boolean using vectorized operations."""
    accuracy_columns = ['Top1_Is_Accurate', 'Top2_Is_Accurate', 'Top3_Is_Accurate']
    data[accuracy_columns] = data[accuracy_columns].astype(bool)
    return data

def plot_rolling_average(data, title):
    """Plot rolling average of confidence scores with legend placement moved further down."""
    with sns.plotting_context("talk"), sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(16, 12))  # Maintaining the adjusted figure size
        metrics = ['Top1', 'Top2', 'Top3']
        accuracies = ['Accurate', 'Inaccurate']

        with sns.color_palette("Paired", n_colors=len(metrics) * 2):
            colors = sns.color_palette()

            for i, metric in enumerate(metrics):
                for j, accuracy in enumerate(accuracies):
                    condition = data[f'{metric}_Is_Accurate'] == (accuracy == 'Accurate')
                    series = data[condition][f'{metric}_Confidence'].rolling(window=100).mean()
                    color_index = i * 2 + (0 if accuracy == 'Inaccurate' else 1)
                    ax.plot(series, label=f'{accuracy} {metric}', color=colors[color_index], alpha=0.8, linewidth=2)

            ax.set_title(f'Confidence Scores Moving Average for {title}', fontsize=16)
            ax.set_xlabel('Predictions', fontsize=14)
            ax.set_ylabel('Confidence Score', fontsize=14)
            # Significantly adjusted legend placement
            ax.legend(title='Prediction Accuracy', title_fontsize='13', fontsize='12', loc='upper center', 
                      bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3)
            plt.tight_layout(rect=[0, 0.2, 1, 0.95])  # Adjusting layout to accommodate the new legend position
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
            plt.show()

# Iterate over datasets, preprocess data, and plot rolling averages
for name, path in datasets.items():
    print(f"Processing dataset: {name}")
    data = pd.read_csv(Path(path))
    data_preprocessed = preprocess_data(data)
    plot_rolling_average(data_preprocessed, name)