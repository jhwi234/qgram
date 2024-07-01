import pandas as pd
import numpy as np
from pathlib import Path
from enum import Enum
import statsmodels.formula.api as smf
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class Letters(Enum):
    VOWELS = 'aeèéiîouyæœ'
    CONSONANTS = 'bcdfghjklmnpqrstvwxz'

def is_vowel(char):
    return char.lower() in Letters.VOWELS.value

def count_vowels(word):
    return sum(1 for char in word if is_vowel(char))

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Missing_Letter_Position'] = data['Tested_Word'].apply(lambda x: x.find('_') if isinstance(x, str) else -1)
    data['Word_Length'] = data['Original_Word'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    data['Normalized_Missing_Letter_Position'] = data.apply(lambda row: row['Missing_Letter_Position'] / (row['Word_Length'] - 1) if row['Word_Length'] > 1 else 0, axis=1)
    data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=10, labels=range(10))
    
    if 'Correct_Letter(s)' in data.columns:
        data['letter_type'] = data['Correct_Letter(s)'].apply(lambda x: 'vowel' if is_vowel(x) else 'consonant')
        data['is_vowel'] = (data['letter_type'] == 'vowel').astype(int)
    else:
        data['is_vowel'] = 0  # Default value if 'Correct_Letter(s)' is missing
    
    # Add number of vowels in the word
    data['Num_Vowels'] = data['Original_Word'].apply(lambda x: count_vowels(x) if isinstance(x, str) else 0)
    
    # Ensure 'Top1_Is_Accurate' is numeric and drop any potential NaN values
    data['Top1_Is_Accurate'] = pd.to_numeric(data['Top1_Is_Accurate'], errors='coerce')
    data = data.dropna(subset=['Top1_Is_Accurate'])
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)

    # Check for consistent lengths
    print(f"Data length after preprocessing: {len(data)}")
    if 'is_vowel' in data.columns:
        print(f"Unique values in 'is_vowel': {data['is_vowel'].unique()}")
    else:
        print(f"'is_vowel' column not found in data")

    return data

def run_logistic_regression(data, dataset_name):
    # Check for sufficient variability
    if len(data['is_vowel'].unique()) == 1:
        print(f"Insufficient variability in 'is_vowel' for {dataset_name} dataset. Skipping regression.")
        return
    
    formula = 'Top1_Is_Accurate ~ is_vowel + Word_Length + Normalized_Missing_Letter_Position + Num_Vowels + Word_Length*Num_Vowels'
    model = smf.logit(formula=formula, data=data).fit()
    print(f"Regression Summary for {dataset_name}:\n")
    print(model.summary())
    
    # Cross-validation with a more complex model
    X = data[['is_vowel', 'Word_Length', 'Normalized_Missing_Letter_Position', 'Num_Vowels']]
    X['Word_Length:Num_Vowels'] = data['Word_Length'] * data['Num_Vowels']
    y = data['Top1_Is_Accurate']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"Cross-Validation Scores for {dataset_name} Random Forest: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores)}\n")
    
    # Feature importance plot
    rf_model.fit(X, y)
    feature_importances = rf_model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance for {dataset_name}')
    plt.show()

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
    }

    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        run_logistic_regression(data, name)

if __name__ == "__main__":
    main()
