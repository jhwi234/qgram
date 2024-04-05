from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings
from pathlib import Path

max_bins = 9

warnings.filterwarnings('ignore', category=FutureWarning)

class Letters(Enum):
    VOWELS = 'aeèéiîouyæœ'
    CONSONANTS = 'bcdfghjklmnpqrstvwxyz'

    @classmethod
    def classify(cls, char):
        if char in cls.VOWELS.value:
            return "Vowel"
        elif char in cls.CONSONANTS.value:
            return "Consonant"
        return "Other"

def preprocess_data(file_path, max_bins=max_bins):
    data = pd.read_csv(file_path)
    data.fillna('', inplace=True)
    data['Missing_Letter_Position'] = data['Tested_Word'].str.find('_')
    data['Word_Length'] = data['Original_Word'].str.len()
    data['Letter_Type'] = [Letters.classify(char) for char in data['Correct_Letter']]
    
    # Ensure there are no entries with Word_Length of zero to avoid division by zero
    data = data[data['Word_Length'] > 0]
    
    # Calculate normalized missing letter position
    data['Normalized_Missing_Letter_Position'] = data['Missing_Letter_Position'] / data['Word_Length']
    
    # Bin the normalized positions into 'max_bins' bins, safely now that we've removed any potential zero lengths
    data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=max_bins, labels=False)

    return data

def plot_accuracy(data, dataset_name):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    for letter_type, color in zip(['Vowel', 'Consonant'], ['#377eb8', '#e41a1c']):
        filtered_data = data[data['Letter_Type'] == letter_type]
        accuracy_summary = filtered_data.groupby('Normalized_Position_Bin')['Top1_Is_Accurate'].mean()
        bin_midpoints = accuracy_summary.index + 0.5
        ax.plot(bin_midpoints, accuracy_summary, '-o', color=color, label=letter_type)
        
        if not accuracy_summary.empty:
            slope, intercept, r_value, _, _ = linregress(bin_midpoints, accuracy_summary)
            ax.plot(bin_midpoints, intercept + slope * bin_midpoints, '--', color=color, alpha=0.5, label=f'{letter_type} Regression: y={intercept:.2f}+{slope:.2f}x')

    ax.set_xlabel('Normalized Missing Letter Position Bin')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title(f'{dataset_name}: Mean Accuracy by Letter Type and Normalized Position')
    ax.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def main():
    dataset_paths = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
    }
    for name, path in dataset_paths.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        plot_accuracy(data, name)

if __name__ == "__main__":
    main()

