from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from scipy import stats

# Define the Letters Enum with methods to check for vowels and consonants
class Letters(Enum):
    VOWELS = 'aeèéiîouyæœ'

    @staticmethod
    def is_vowel(char):
        return char in Letters.VOWELS.value

    @staticmethod
    def is_consonant(char):
        return char.isalpha() and not Letters.is_vowel(char)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Ensure all entries are treated as strings
    data['Original_Word'] = data['Original_Word'].astype(str)
    data['Tested_Word'] = data['Tested_Word'].astype(str)

    # Find the position of the missing letter
    data['Missing_Letter_Position'] = data['Tested_Word'].apply(lambda x: x.find('_') if '_' in x else -1)

    # Calculate the length of the original word
    data['Word_Length'] = data['Original_Word'].apply(len)

    # Extract the missing letter based on its position and classify it using the Letters Enum
    data['Missing_Letter'] = data.apply(
        lambda row: row['Original_Word'][row['Missing_Letter_Position']]
        if 0 <= row['Missing_Letter_Position'] < len(row['Original_Word']) else '', axis=1)
    
    data['Letter_Type'] = data['Missing_Letter'].apply(
        lambda x: 'Vowel' if Letters.is_vowel(x) else 'Consonant' if Letters.is_consonant(x) else 'None')

    # Normalize the position of the missing letter and bin it
    data['Normalized_Missing_Letter_Position'] = data.apply(
        lambda row: row['Missing_Letter_Position'] / (row['Word_Length'] - 1) if row['Word_Length'] > 1 else 0, axis=1)
    data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=10, labels=range(10))

    return data

def plot_line_with_confidence_intervals_and_regression(data, dataset_name, letter_type):
    # Filter data based on the letter type (Vowel or Consonant)
    filtered_data = data[data['Letter_Type'] == letter_type]
    accuracy_summary = filtered_data.groupby('Normalized_Position_Bin')['Top1_Is_Accurate'].agg(['mean', 'std', 'count'])
    accuracy_summary['se'] = accuracy_summary['std'] / np.sqrt(accuracy_summary['count'])
    accuracy_summary['ci'] = 1.96 * accuracy_summary['se']
    accuracy_summary['Bin_Midpoint'] = accuracy_summary.index.astype(float) + 0.5
    accuracy_summary.reset_index(drop=True, inplace=True)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(accuracy_summary['Bin_Midpoint'], accuracy_summary['mean'])

    plt.figure(figsize=(12, 8), dpi=100)
    plt.errorbar(x=range(10), y=accuracy_summary['mean'], yerr=accuracy_summary['ci'], fmt='-o', capsize=5, color='#377eb8', ecolor='lightgray', elinewidth=3, capthick=2, markersize=5)
    plt.plot(accuracy_summary['Bin_Midpoint'], intercept + slope * accuracy_summary['Bin_Midpoint'], '--', label=f'Regression Line: $y={intercept:.4f}+{slope:.4f}x$, $R^2={r_value**2:.4f}$', color='#e41a1c')

    plt.xlabel('Normalized Missing Letter Position Bin', fontsize=14)
    plt.ylabel('Mean Accuracy', fontsize=14)
    plt.title(f'{dataset_name} ({letter_type}): Mean Accuracy vs. Normalized Missing Letter Position', fontsize=16)
    plt.xticks(range(10), labels=[f"Bin {i}" for i in range(10)])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

    letter_types = ['Vowel', 'Consonant']

    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        for letter_type in letter_types:
            print(f"\nPlotting for {letter_type}s...")
            plot_line_with_confidence_intervals_and_regression(data, name, letter_type)

if __name__ == "__main__":
    main()
