import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import Parallel, delayed

# Apply settings for all plots
sns.set(style="whitegrid", context="notebook", palette="Paired")

# Define dataset paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

def get_color_palette():
    """ Generates a reversed color palette from seaborn's 'Paired' palette. """
    base_colors = sns.color_palette("Paired", 12)
    return [[base_colors[i + 1], base_colors[i]] for i in range(0, len(base_colors), 2)]

def parallel_bootstrap(data, n_bootstrap=1000, ci=95):
    """ Uses parallel processing to efficiently compute bootstrap confidence intervals. """
    if data.empty:
        return None, None, None  # Handle empty data gracefully

    def bootstrap_sample(sample_data):
        return np.mean(np.random.choice(sample_data, len(sample_data), replace=True))

    results = Parallel(n_jobs=-1)(delayed(bootstrap_sample)(data) for _ in range(n_bootstrap))
    confidence_bounds = np.percentile(results, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    mean_confidence = np.mean(results)
    return confidence_bounds[0], confidence_bounds[1], mean_confidence

def plot_confidence_intervals(data, title):
    """ Plots confidence intervals for the given data using a bar chart. """
    colors = get_color_palette()
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['Top1', 'Top2', 'Top3']
    accuracies = ['Accurate', 'Inaccurate']

    bar_width = 0.35
    index = np.arange(len(metrics) * 2)

    for i, metric in enumerate(metrics):
        for j, accuracy in enumerate(accuracies):
            series = data[data[f'{metric}_Is_Accurate'] == (accuracy == 'Accurate')][f'{metric}_Confidence']
            lower, upper, mean_confidence = parallel_bootstrap(series)
            if lower is not None:
                ax.bar(index[i * 2 + j], mean_confidence, bar_width, color=colors[i][j], label=f'{accuracy} {metric}')
                ax.text(index[i * 2 + j], mean_confidence, f'{mean_confidence:.2%}', ha='center', va='bottom')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Mean Confidence')
    ax.set_title(f'Confidence Intervals for {title}')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{acc} {met}' for met in metrics for acc in accuracies], rotation=45)
    ax.legend(title="Metric Accuracy", loc='upper right', fontsize='medium', title_fontsize='large')
    plt.tight_layout()
    plt.show()

def preprocess_data(data):
    """ Converts accuracy indicators to boolean for all Top metrics. """
    accuracy_columns = ['Top1_Is_Accurate', 'Top2_Is_Accurate', 'Top3_Is_Accurate']
    data[accuracy_columns] = data[accuracy_columns].astype(bool)
    return data

# Processing datasets
for name, path in datasets.items():
    print(f"Processing dataset: {name}")
    try:
        data = pd.read_csv(Path(path))
        if not data.empty:
            data_preprocessed = preprocess_data(data)
            plot_confidence_intervals(data_preprocessed, name)
        else:
            print("No data to process after preprocessing.")
    except Exception as e:
        print(f"Error processing {name}: {e}")
