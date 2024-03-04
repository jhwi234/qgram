from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Adjust the file path to navigate up one directory and then into the 'data' directory
file_path = Path('data/outputs/csv/brown_context_sensitive_HAPAX_split0.5_qrange6-6_prediction.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Define vowels
vowels = ['a', 'e', 'i', 'o', 'u', 'æ', 'œ']

# Classify the correct letters as vowel or consonant
data['letter_type'] = data['Correct_Letter'].apply(lambda x: 'vowel' if x.lower() in vowels else 'consonant')

# Encode 'letter_type' as a categorical variable
data['is_vowel'] = (data['letter_type'] == 'vowel').astype(int)

# Conducting logistic regression analysis with 'is_vowel' as a predictor
model = smf.logit(formula='Top1_Is_Accurate ~ is_vowel', data=data).fit()

# Displaying the regression results
print(model.summary())
