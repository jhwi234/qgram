import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path  # Importing Path class

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
    elif char in "wy":  # Added 'y' to Glide as it was previously considered only as a vowel
        return 'Glide'
    elif is_vowel(char):
        return 'Vowel'
    else:
        return 'Other'

def preprocess_data_simple(df):
    df['is_vowel'] = df['Correct_Letter'].apply(lambda x: 1 if is_vowel(x) else 0)
    df['Phonological_Category'] = df['Correct_Letter'].apply(classify_phonological_category)
    df = pd.get_dummies(df, columns=['Phonological_Category'], drop_first=True)
    return df

def run_logistic_regression_corrected(df):
    model = smf.logit(formula='Top1_Is_Accurate ~ is_vowel', data=df).fit()
    print(model.summary())

def run_logistic_regression_phonological(df):
    phonological_categories = [col for col in df.columns if col.startswith('Phonological_Category_')]
    formula = 'Top1_Is_Accurate ~ ' + ' + '.join(phonological_categories)
    model = smf.logit(formula=formula, data=df).fit()
    print(model.summary())

def main():
    datasets = {
        "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv',
        "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv',
        "CMUDict": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'
    }

    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        df = pd.read_csv(path)
        df_preprocessed = preprocess_data_simple(df)
        print(f"\nResults for Corrected Predictions in {name}:")
        run_logistic_regression_corrected(df_preprocessed)
        print(f"\nResults for Phonological Predictions in {name}:")
        run_logistic_regression_phonological(df_preprocessed)

if __name__ == "__main__":
    main()
