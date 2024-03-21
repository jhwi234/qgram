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
    df = pd.get_dummies(df, columns=['Phonological_Category'], drop_first=False)
    return df

# Logistic regression analysis with phonological categories
def run_logistic_regression_phonological_adjusted(df):
    # Create a list of phonological category columns, excluding 'Vowel'
    phonological_categories = [col for col in df.columns if col.startswith('Phonological_Category_') and 'Vowel' not in col]
    # Construct the formula, ensuring 'Vowel' is the reference by its absence
    formula = f'Top1_Is_Accurate ~ {" + ".join(phonological_categories)}'
    # Fit the logistic regression model with L1 regularization and a high maxiter
    model = smf.logit(formula=formula, data=df).fit_regularized(method='l1', maxiter=1000)
    print(model.summary())

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
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
