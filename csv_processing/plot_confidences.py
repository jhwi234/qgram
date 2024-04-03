from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

# Setting a color-blind friendly style
plt.style.use('seaborn-v0_8-colorblind')

# Define the datasets paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv'
}

# Color configurations
colors = {
    'accurate_top1': '#E69F00',
    'inaccurate_top1': '#56B4E9',
    'accurate_top2': '#009E73',
    'inaccurate_top2': '#F0E442',
    'accurate_top3': '#0072B2',
    'inaccurate_top3': '#D55E00',
}

window_size = 100  # Number of instances for the moving average

def plot_dataset(name, path):
    print(f"Processing dataset: {name}")
    data = pd.read_csv(path)

    # Preprocessing for optimization
    for rank in ['Top1', 'Top2', 'Top3']:
        data[f'{rank}_Is_Accurate'] = data[f'{rank}_Is_Accurate'].astype('bool')

    metrics = {}
    # Calculating moving averages for metrics
    for rank in ['Top1', 'Top2', 'Top3']:
        for accuracy in ['Accurate', 'Inaccurate']:
            key = f'{accuracy.lower()}_{rank.lower()}'
            condition = data[f'{rank}_Is_Accurate'] == (accuracy == 'Accurate')
            metrics[key] = data[condition][f'{rank}_Confidence'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 6))
    legend_elements = []

    for metric, series in metrics.items():
        accuracy, rank = metric.split('_')
        color = colors[metric]
        plt.plot(series, label=f'{accuracy.capitalize()} {rank.upper()}', color=color, alpha=0.9)
        
        # Create a legend element with a square marker
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10,
                                      label=f'{accuracy.capitalize()} {rank.upper()}'))

    plt.title(f'Confidence Scores for {name}', fontsize=14)
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Confidence Score (Moving Average)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

for name, path in datasets.items():
    plot_dataset(name, Path(path))
