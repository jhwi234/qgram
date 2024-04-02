from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Using a predefined style
plt.style.use('seaborn-v0_8-colorblind')  # This is a color-blind friendly style

# Define the datasets paths
datasets = {
    # Ensure these paths are correct for your setup
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
}

# Color and line style configurations
colors = {
    'accurate_top1': '#E69F00',
    'inaccurate_top1': '#56B4E9',
    'accurate_top2': '#009E73',
    'inaccurate_top2': '#F0E442',
    'accurate_top3': '#0072B2',
    'inaccurate_top3': '#D55E00',
}

line_styles = {
    'accurate': '-',
    'inaccurate': ':'
}

window_size = 50  # Number of instances for the moving average

# Iterating through each dataset to create individual plots
for name, path in datasets.items():
    print(f"Processing dataset: {name}")
    data = pd.read_csv(path)

    # Calculating moving averages for metrics
    metrics = {}
    for rank in ['Top1', 'Top2', 'Top3']:
        for accuracy in ['Accurate', 'Inaccurate']:
            key = f'{accuracy.lower()}_{rank.lower()}'
            condition = data[f'{rank}_Is_Accurate'] == (accuracy == 'Accurate')
            simple_series = data[condition][f'{rank}_Confidence'].rolling(window=window_size).mean()
            metrics[key] = simple_series

    # Creating individual plots
    plt.figure(figsize=(10, 6))
    for metric, series in metrics.items():
        accuracy, rank = metric.split('_')
        plt.plot(series, label=f'{accuracy.capitalize()} {rank.upper()} (Smoothed)',
                 color=colors[metric], alpha=0.9, linestyle=line_styles[accuracy])

    # Enhancing visual appearance
    plt.title(f'Smoothed Confidence Scores for {name}', fontsize=14)
    plt.xlabel('Instance Index', fontsize=12)
    plt.ylabel('Confidence Score (Moving Average)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Positioning the legend outside the plot area in the center of the figure's bottom
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the legend
    plt.show()