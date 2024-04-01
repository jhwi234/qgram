from pathlib import Path
import pandas as pd
from enum import Enum
import statsmodels.formula.api as smf

class Letters(Enum):
    VOWELS = 'aeèéiîouyæœ'
    CONSONANTS = 'bcdfghjklmnpqrstvwxz'

    @staticmethod
    def is_vowel(char):
        return char in Letters.VOWELS.value

    @staticmethod
    def is_consonant(char):
        return char in Letters.CONSONANTS.value

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Missing_Letter_Position'] = data['Tested_Word'].apply(lambda x: x.find('_') if isinstance(x, str) else -1)
    data['Word_Length'] = data['Original_Word'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    data['Normalized_Missing_Letter_Position'] = data.apply(lambda row: row['Missing_Letter_Position'] / (row['Word_Length'] - 1) if row['Word_Length'] > 1 else 0, axis=1)
    data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=10, labels=range(10))
    if 'Correct_Letter' in data.columns:
        data['letter_type'] = data['Correct_Letter'].apply(lambda x: 'vowel' if Letters.is_vowel(x.lower()) else 'consonant')
        data['is_vowel'] = (data['letter_type'] == 'vowel').astype(int)
    return data

def run_logistic_regression(data, dataset_name):
    if 'is_vowel' in data.columns:
        model = smf.logit(formula='Top1_Is_Accurate ~ is_vowel', data=data).fit()
        print(f"Regression Summary for {dataset_name}:\n")
        print(model.summary())
    else:
        print(f"Data preprocessing was not completed properly for {dataset_name}, or the 'Correct_Letter' column is missing.")

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        run_logistic_regression(data, name)

if __name__ == "__main__":
    main()
