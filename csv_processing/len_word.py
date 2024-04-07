from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Define a function to perform logistic regression analysis with Top1_Is_Accurate as a boolean
def analyze_word_length_accuracy(file_path):
    data = pd.read_csv(file_path)

    # Handle missing values by treating them as empty strings for word length calculation
    data['Word_Length'] = data['Original_Word'].fillna('').apply(len)
    
    # Ensure 'Top1_Is_Accurate' is boolean, then convert True/False to 1/0
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    
    # Calculate additional statistical measures
    word_length_stats = data.groupby('Word_Length').agg({
        'Top1_Is_Accurate': ['mean', 'std', 'count', 'sem']
    }).reset_index()
    word_length_stats.columns = ['Word_Length', 'Accuracy_Mean', 'Accuracy_Std', 'Sample_Count', 'Accuracy_SEM']

    print("Statistical Analysis of Word Length and Prediction Accuracy:")
    print(word_length_stats)

    # Check data types before regression
    print("\nData Types:")
    print(data.dtypes)

    # Perform logistic regression with word length as the predictor of top prediction accuracy
    logistic_regression_model = smf.logit('Top1_Is_Accurate ~ Word_Length', data=data).fit()

    print("\nLogistic Regression Analysis Summary:")
    print(logistic_regression_model.summary())

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
    print(f"\n{dataset_name} Dataset Analysis:")
    analyze_word_length_accuracy(file_path)
