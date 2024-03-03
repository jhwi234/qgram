from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Adjust the file path to navigate up one directory and then into the 'data' directory
file_path = Path('../data/outputs/csv/brown_context_sensitive_HAPAX_split0.5_qrange6-6_prediction.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Ensuring all relevant fields are treated as strings
data['Original_Word'] = data['Original_Word'].astype(str)
data['Tested_Word'] = data['Tested_Word'].astype(str)
data['Top1_Predicted_Letter'] = data['Top1_Predicted_Letter'].astype(str)

# Calculate word length
data['word_length'] = data['Original_Word'].apply(len)

# Perform logistic regression to investigate the relationship between word length and Top1_Is_Accurate
model = smf.logit(formula='Top1_Is_Accurate ~ word_length', data=data).fit()

# Display the regression results
print(model.summary())