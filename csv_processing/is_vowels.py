import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# Define the function to perform logistic regression analysis on each dataset
def logistic_regression_analysis_vowel_effect(file_path, dataset_name):
    data = pd.read_csv(file_path)

    # Ensure 'Top1_Is_Accurate' is boolean and convert to integer (1 for True, 0 for False)
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)

    # Determine if the predicted letter is a vowel
    vowels = 'aeiou'
    data['Is_Vowel'] = data['Top1_Predicted_Letter'].apply(lambda x: 1 if x in vowels else 0)

    # Fit logistic regression model
    model = sm.Logit(data['Top1_Is_Accurate'], sm.add_constant(data['Is_Vowel'])).fit()

    print(f"\n{dataset_name} Dataset Analysis:")
    print(model.summary2())
    return model

# Define dataset paths
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Perform analysis for each dataset
models = {}
for dataset_name, file_path in dataset_paths.items():
    print(f"Running analysis for {dataset_name} dataset...")
    models[dataset_name] = logistic_regression_analysis_vowel_effect(file_path, dataset_name)
