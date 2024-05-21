import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# Define the function to perform logistic regression analysis on each dataset
def logistic_regression_analysis(file_path, dataset_name):
    data = pd.read_csv(file_path)

    # Calculate the word length
    data['Word_Length'] = data['Original_Word'].apply(len)
    
    # Calculate the relative position of the missing letter
    data['Relative_Position'] = data['Correct_Letter_Rank'] / data['Word_Length']

    # Ensure 'Top1_Is_Accurate' is boolean and convert to integer (1 for True, 0 for False)
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)

    # Fit logistic regression model
    model = smf.logit('Top1_Is_Accurate ~ Relative_Position', data=data).fit()

    print(f"\n{dataset_name} Dataset Analysis:")
    print(model.summary())

# Dictionary mapping dataset names to their file paths
datasets = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Perform analysis for each dataset
for dataset_name, file_path in datasets.items():
    logistic_regression_analysis(file_path, dataset_name)