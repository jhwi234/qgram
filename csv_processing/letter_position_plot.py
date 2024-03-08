import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Missing_Letter_Position'] = data['Tested_Word'].apply(lambda x: x.find('_') if isinstance(x, str) else -1)
    data['Word_Length'] = data['Original_Word'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    data['Normalized_Missing_Letter_Position'] = data.apply(lambda row: row['Missing_Letter_Position'] / (row['Word_Length'] - 1) if row['Word_Length'] > 1 else 0, axis=1)
    data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=10, labels=range(10))
    return data

def plot_line_with_confidence_intervals_and_regression(data, dataset_name):
    accuracy_summary = data.groupby('Normalized_Position_Bin')['Top1_Is_Accurate'].agg(['mean', 'std', 'count'])
    accuracy_summary['se'] = accuracy_summary['std'] / np.sqrt(accuracy_summary['count'])
    accuracy_summary['ci'] = 1.96 * accuracy_summary['se']
    accuracy_summary['Bin_Midpoint'] = accuracy_summary.index.astype(float) + 0.5
    accuracy_summary.reset_index(inplace=True)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(accuracy_summary['Bin_Midpoint'], accuracy_summary['mean'])
    
    plt.figure(figsize=(12, 8), dpi=100)
    plt.errorbar(x=accuracy_summary['Normalized_Position_Bin'], y=accuracy_summary['mean'], yerr=accuracy_summary['ci'], fmt='-o', capsize=5, color='#377eb8', ecolor='lightgray', elinewidth=3, capthick=2, markersize=5)
    plt.plot(accuracy_summary['Bin_Midpoint'], intercept + slope*accuracy_summary['Bin_Midpoint'], '--', label=f'Regression Line: $y={intercept:.2f}+{slope:.2f}x$, $R^2={r_value**2:.2f}$', color='#e41a1c')
    
    plt.xlabel('Normalized Missing Letter Position Bin', fontsize=14)
    plt.ylabel('Mean Accuracy', fontsize=14)
    plt.title(f'{dataset_name}: Mean Accuracy vs. Normalized Missing Letter Position', fontsize=16)
    plt.xticks(range(10), labels=[f"Bin {i}" for i in range(10)])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "CMUDict": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        plot_line_with_confidence_intervals_and_regression(data, name)

if __name__ == "__main__":
    main()
