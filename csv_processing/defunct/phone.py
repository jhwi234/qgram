import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

def is_vowel(char):
    vowels = 'aeiouyæœèéî'
    return char.lower() in vowels

def classify_phonological_category(char):
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

def preprocess_data_adjusted(df):
    df['Phonological_Category'] = df['Correct_Letter'].apply(classify_phonological_category)
    # Attempt to convert 'Top1_Is_Accurate' to numeric, coercing errors
    df['Top1_Is_Accurate'] = pd.to_numeric(df['Top1_Is_Accurate'], errors='coerce')
    # Drop rows with NaN values in 'Top1_Is_Accurate'
    df.dropna(subset=['Top1_Is_Accurate'], inplace=True)
    # Ensure 'Top1_Is_Accurate' is integer
    df['Top1_Is_Accurate'] = df['Top1_Is_Accurate'].astype(int)
    return df

def run_logistic_regression_phonological_adjusted(df):
    print("Data types before regression:", df.dtypes)  # Debugging line
    formula = 'Top1_Is_Accurate ~ C(Phonological_Category, Treatment(reference="Vowel"))'
    model = smf.logit(formula, data=df).fit()
    print(model.summary())

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
        df = pd.read_csv(path)
        df_preprocessed = preprocess_data_adjusted(df)
        print(f"\nResults for Phonological Predictions in {name}:")
        run_logistic_regression_phonological_adjusted(df_preprocessed)

if __name__ == "__main__":
    main()