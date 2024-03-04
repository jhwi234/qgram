from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Adjust the file path appropriately
file_path = Path('data/outputs/csv/brown_context_sensitive_HAPAX_split0.5_qrange6-6_prediction.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Ensure all relevant columns are treated as strings
data['Original_Word'] = data['Original_Word'].astype(str)
data['Tested_Word'] = data['Tested_Word'].astype(str)

# Calculate the missing position and relative position with zero-based indexing in mind
data['missing_position'] = data['Tested_Word'].str.find('_')
data['word_length'] = data['Original_Word'].str.len()
data['relative_position'] = data['missing_position'] / (data['word_length'] - 1)

# Define a function to categorize positions
def categorize_position(row):
    if row['missing_position'] == 0:
        return 'first'
    elif row['missing_position'] == row['word_length'] - 1:
        return 'last'
    elif row['word_length'] % 2 == 1 and row['missing_position'] == row['word_length'] // 2:
        return 'middle'
    elif row['word_length'] % 2 == 0 and (row['missing_position'] == row['word_length'] // 2 or row['missing_position'] == (row['word_length'] // 2 - 1)):
        return 'middle'
    else:
        return 'medial'

# Apply the function to the data
data['position_category'] = data.apply(categorize_position, axis=1)

# Create dummy variables for categorical analysis of letter position
data = pd.get_dummies(data, columns=['position_category'], prefix='', prefix_sep='')

# Aggregating accuracy based on relative positions for regression analysis
relative_position_accuracy = data.groupby('relative_position')['Top1_Is_Accurate'].mean().reset_index(name='average_accuracy')

# Performing regression analysis for relative position
model_rel_pos = smf.ols('average_accuracy ~ relative_position', data=relative_position_accuracy).fit()

# For categorical positions (first, middle, last, medial), we include them as predictors in a single model
model_categorical = smf.ols(
    'Top1_Is_Accurate ~ first + middle + last + medial',
    data=data
).fit()

# Displaying the regression results
print("Regression Analysis for Relative Position:")
print(model_rel_pos.summary())

print("\nRegression Analysis for Categorical Letter Positions:")
print(model_categorical.summary())
