from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
class Letters(Enum):
    VOWELS = 'aeèéiîouyæœ'

    @staticmethod
    def is_vowel(char):
        return char in Letters.VOWELS.value

    @staticmethod
    def is_consonant(char):
        return char.isalpha() and not Letters.is_vowel(char)

def classify_letter(char):
    if Letters.is_vowel(char):
        return "Vowel"
    elif Letters.is_consonant(char):
        return "Consonant"
    else:
        return "Other"

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Original_Word'] = data['Original_Word'].astype(str)
    data['Tested_Word'] = data['Tested_Word'].astype(str)
    data['Missing_Letter_Position'] = data['Tested_Word'].apply(lambda x: x.find('_') if '_' in x else -1)
    data['Word_Length'] = data['Original_Word'].apply(len)
    data['Missing_Letter'] = data.apply(lambda row: row['Original_Word'][row['Missing_Letter_Position']]
                                        if 0 <= row['Missing_Letter_Position'] < len(row['Original_Word']) else '', axis=1)
    data['Letter_Type'] = data['Missing_Letter'].apply(classify_letter)
    data['Normalized_Missing_Letter_Position'] = data.apply(lambda row: row['Missing_Letter_Position'] / (row['Word_Length'] - 1)
                                                             if row['Word_Length'] > 1 else 0, axis=1)
    data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=10, labels=range(10))
    return data

def plot_vowel_consonant_comparison_connected(data, dataset_name):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    colors = ['#377eb8', '#e41a1c']  # Blue for vowels, red for consonants
    markers = ['o', 's']  # Circle for vowels, square for consonants
    letter_types = ['Vowel', 'Consonant']

    for color, marker, letter_type in zip(colors, markers, letter_types):
        filtered_data = data[data['Letter_Type'] == letter_type]
        accuracy_summary = filtered_data.groupby('Normalized_Position_Bin')['Top1_Is_Accurate'].agg(['mean', 'std', 'count'])
        accuracy_summary['se'] = accuracy_summary['std'] / np.sqrt(accuracy_summary['count'])
        accuracy_summary['ci'] = 1.96 * accuracy_summary['se']
        
        # Calculate the bin midpoints correctly for centered plotting
        bin_edges = np.linspace(0, 1, 11)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        accuracy_summary.reset_index(drop=True, inplace=True)
        ax.errorbar(x=bin_midpoints, y=accuracy_summary['mean'], yerr=accuracy_summary['ci'], fmt=marker,
                    capsize=5, color=color, ecolor='lightgray', elinewidth=3, capthick=2, markersize=5, label=letter_type)
        ax.plot(bin_midpoints, accuracy_summary['mean'], linestyle='-', color=color)
    
    ax.set_xlabel('Normalized Missing Letter Position Bin', fontsize=14)
    ax.set_ylabel('Mean Accuracy', fontsize=14)
    ax.set_title(f'{dataset_name}: Mean Accuracy by Letter Type and Normalized Position', fontsize=16)
    ax.set_xticks(bin_midpoints)
    ax.set_xticklabels([f"Bin {i}" for i in range(10)])
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }
    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        plot_vowel_consonant_comparison_connected(data, name)

if __name__ == "__main__":
    main()

