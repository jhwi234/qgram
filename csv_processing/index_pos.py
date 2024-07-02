import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import numpy as np

# Paths to datasets
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Function to preprocess data
def preprocess_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if not {'Tested_Word', 'Top1_Is_Accurate'}.issubset(data.columns):
        raise ValueError("Required columns are missing from the dataset")
    
    # Calculate word length and normalized missing letter position
    data['Word_Length'] = data['Tested_Word'].str.len()
    data['Normalized_Missing_Letter_Position'] = data['Tested_Word'].apply(
        lambda word: word.index('_') / (len(word) - 1) if '_' in word else np.nan
    )
    
    return data

# Function to perform logistic regression analysis
def logistic_regression_analysis(file_path: Path, dataset_name: str):
    data = preprocess_data(file_path)
    
    # Ensure 'Top1_Is_Accurate' is boolean and convert to integer (1 for True, 0 for False)
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    
    # Drop rows with NaN values in 'Normalized_Missing_Letter_Position'
    data = data.dropna(subset=['Normalized_Missing_Letter_Position'])
    
    # Fit logistic regression model
    model = smf.logit('Top1_Is_Accurate ~ Normalized_Missing_Letter_Position', data=data).fit()
    
    print(f"\n{dataset_name} Dataset Analysis:")
    print(model.summary())

# Perform analysis for each dataset
for dataset_name, file_path in dataset_paths.items():
    logistic_regression_analysis(file_path, dataset_name)
