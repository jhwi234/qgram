from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Define a function to perform robust analysis
def analyze_word_length_accuracy(file_path):
    data = pd.read_csv(file_path)

    # Handle missing values by treating them as empty strings for word length calculation
    data['Word_Length'] = data['Original_Word'].fillna('').apply(len)
    
    # Calculate additional statistical measures
    word_length_stats = data.groupby('Word_Length').agg({
        'Top1_Is_Accurate': ['mean', 'std', 'count', 'sem']
    }).reset_index()
    word_length_stats.columns = ['Word_Length', 'Accuracy_Mean', 'Accuracy_Std', 'Sample_Count', 'Accuracy_SEM']

    print("Statistical Analysis of Word Length and Prediction Accuracy:")
    print(word_length_stats)

    # Perform robust linear regression with word length as the predictor of top prediction accuracy
    robust_regression_model = smf.rlm('Top1_Is_Accurate ~ Word_Length', data=data).fit()

    print("\nRobust Regression Analysis Summary:")
    print(robust_regression_model.summary())

# File paths
clmet_file_path = Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv')
brown_file_path = Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
cmudict_file_path = Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv')

# Perform analysis for each dataset
print("CLMET3 Dataset Analysis:")
analyze_word_length_accuracy(clmet_file_path)

print("\nBrown Dataset Analysis:")
analyze_word_length_accuracy(brown_file_path)

print("\nCMUDict Dataset Analysis:")
analyze_word_length_accuracy(cmudict_file_path)