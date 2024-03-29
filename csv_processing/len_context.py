from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Dictionary mapping dataset names to their file paths
datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

# Select and load a specific dataset
file_path = datasets["CLMET3"]
data = pd.read_csv(file_path)

# Ensure specific columns are treated as strings
data['Original_Word'] = data['Original_Word'].astype(str)
data['Tested_Word'] = data['Tested_Word'].astype(str)
data['Top1_Predicted_Letter'] = data['Top1_Predicted_Letter'].astype(str)

# Calculate word length and context length around the missing letter
data['word_length'] = data['Original_Word'].apply(len)
data['context_length_left'] = data['Tested_Word'].str.find('_')
data['context_length_right'] = data['word_length'] - data['context_length_left'] - 1
data['context_length'] = data['context_length_left'] + data['context_length_right']

# Logistic regression: context length vs. accuracy of the top-1 predicted letter
model = smf.logit(formula='Top1_Is_Accurate ~ context_length', data=data).fit()

# Display regression analysis summary
print(model.summary())
