from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Adjust the file path to navigate up one directory and then into the 'data' directory
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

# Categorize positions as first, middle, last considering zero-based indexing
data['position_category'] = np.select(
    [
        data['missing_position'] == 0,
        data['missing_position'] == data['word_length'] - 1
    ],
    [
        'first',
        'last'
    ],
    default='middle'
)

# Create dummy variables for categorical analysis of letter position
data = pd.get_dummies(data, columns=['position_category'])

# Aggregating accuracy based on relative positions for regression analysis
relative_position_accuracy = data.groupby('relative_position')['Top1_Is_Accurate'].mean().reset_index(name='average_accuracy')

# Performing regression analysis for relative position
model_rel_pos = smf.ols('average_accuracy ~ relative_position', data=relative_position_accuracy).fit()

# For categorical positions (first, middle, last), we include them as predictors in a single model
model_categorical = smf.ols(
    'Top1_Is_Accurate ~ position_category_first + position_category_middle + position_category_last + relative_position',
    data=data
).fit()

# Displaying the regression results
print("Regression Analysis for Relative Position:")
print(model_rel_pos.summary())

print("\nRegression Analysis for Categorical Letter Positions and Relative Position:")
print(model_categorical.summary())
