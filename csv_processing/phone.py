import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# Define functions for phonological category classification
def is_vowel(char):
    vowels = 'aeiouyæœèéî'
    return char.lower() in vowels

def classify_phonological_category(char):
    char = char.lower()
    if char in "pbtdkg":
        return 'Plosive'
    elif char in "mn":
        return 'Nasal'
    elif char in "fvsz":
        return 'Fricative'
    elif char in "lr":
        return 'Liquid'
    elif is_vowel(char):
        return 'Vowel'
    else:
        return 'Other'

# Preprocess the data
def preprocess_data_adjusted(df):
    df['Phonological_Category'] = df['Correct_Letter'].apply(classify_phonological_category)
    return df

# Logistic regression analysis with phonological categories
def run_logistic_regression_phonological_adjusted(df):
    # Specify 'Vowel' as the reference category using C(variable, Treatment(reference='Vowel'))
    formula = 'Top1_Is_Accurate ~ C(Phonological_Category, Treatment(reference="Vowel"))'
    # Fit the logistic regression model
    model = smf.logit(formula, data=df).fit()
    print(model.summary())

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
        df = pd.read_csv(path)
        df_preprocessed = preprocess_data_adjusted(df)
        print(f"\nResults for Phonological Predictions in {name}:")
        run_logistic_regression_phonological_adjusted(df_preprocessed)

if __name__ == "__main__":
    main()
