from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Adjust the file path to navigate up one directory and then into the 'data' directory
file_path = Path('data/outputs/csv/CLMET3_context_sensitive_HAPAX_split0.5_qrange6-6_prediction.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Calculate the total length of each original word
data['Word_Length'] = data['Original_Word'].apply(len)

# Group by word length and calculate mean accuracy
word_length_accuracy = data.groupby('Word_Length')['Top1_Is_Accurate'].mean().reset_index()

print("Word Length vs. Prediction Accuracy:")
print(word_length_accuracy)

# Perform linear regression with word length as the predictor of top prediction accuracy
regression_model = smf.ols('Top1_Is_Accurate ~ Word_Length', data=data).fit()

# Display the regression summary to examine the coefficients, R-squared value, and other statistics
print("\nRegression Analysis Summary:")
print(regression_model.summary())