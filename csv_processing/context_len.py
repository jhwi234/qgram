from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Adjust the file path to navigate up one directory and then into the 'data' directory
file_path = Path('data/outputs/csv/CLMET3_context_sensitive_HAPAX_split0.5_qrange6-6_prediction.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Ensuring all relevant fields are treated as strings
data['Original_Word'] = data['Original_Word'].astype(str)
data['Tested_Word'] = data['Tested_Word'].astype(str)
data['Top1_Predicted_Letter'] = data['Top1_Predicted_Letter'].astype(str)

# Calculate word length
data['word_length'] = data['Original_Word'].apply(len)

# Calculate context length on the left and right
data['context_length_left'] = data['Tested_Word'].str.find('_')
data['context_length_right'] = data['word_length'] - data['context_length_left'] - 1

# Combine the context lengths into a single variable
data['context_length'] = data['context_length_left'] + data['context_length_right']

# Perform logistic regression to investigate the relationship between context length and Top1_Is_Accurate
model = smf.logit(formula='Top1_Is_Accurate ~ context_length', data=data).fit()

# Display the regression results
print(model.summary())
