import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Define function to check if a letter is a vowel
def is_vowel(letter):
    vowels = 'aeiou'
    return letter.lower() in vowels

# Preprocess the data
def preprocess_data(data):
    # Calculate the word length
    data['Word_Length'] = data['Original_Word'].apply(len)
    
    # Calculate the relative position of the missing letter
    data['Missing_Letter_Position'] = data['Tested_Word'].apply(lambda x: x.index('_') if '_' in x else -1)
    data['Relative_Position'] = data['Missing_Letter_Position'] / data['Word_Length']
    
    # Determine if the missing letter is a vowel or consonant
    data['Missing_Letter'] = data.apply(lambda row: row['Original_Word'][row['Missing_Letter_Position']], axis=1)
    data['Is_Vowel'] = data['Missing_Letter'].apply(is_vowel).astype(int)
    
    # Ensure 'Top1_Is_Accurate' is boolean and convert to integer (1 for True, 0 for False)
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    
    return data

# Dictionary mapping dataset names to their file paths
datasets = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Function to run the logistic regression analysis on each dataset
def logistic_regression_analysis(dataset_name, file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Select features and target variable
    X = data[['Relative_Position', 'Word_Length', 'Is_Vowel']]
    y = data['Top1_Is_Accurate']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and fit the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Coefficients of the model
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Print the evaluation metrics and model coefficients
    print(f'\n{dataset_name} Dataset Analysis:')
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
    print('Model Coefficients:')
    print(f'Relative Position: {coefficients[0]:.4f}')
    print(f'Word Length: {coefficients[1]:.4f}')
    print(f'Is Vowel: {coefficients[2]:.4f}')
    print(f'Intercept: {intercept:.4f}')

# Perform analysis for each dataset
for dataset_name, file_path in datasets.items():
    logistic_regression_analysis(dataset_name, file_path)
