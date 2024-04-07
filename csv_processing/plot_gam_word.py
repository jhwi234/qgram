from pygam import LogisticGAM, s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to your datasets
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

for name, path in dataset_paths.items():
    print(f"Processing {name} dataset...")
    # Load the dataset
    data = pd.read_csv(path)

    # Calculate word length if not already a column
    if 'Word_Length' not in data.columns:
        data['Word_Length'] = data['Original_Word'].apply(len)
    
    # Ensure the accuracy column is binary and suitable for logistic regression
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)

    # Define the independent variable (word length) and dependent variable (accuracy)
    X = data[['Word_Length']]
    y = data['Top1_Is_Accurate']

    # Fit a GAM with a logistic link function for binary classification
    gam = LogisticGAM(s(0, n_splines=25)).fit(X, y)

    # Generating predictions for plotting
    XX = np.linspace(X.min(), X.max(), 500)
    proba = gam.predict_proba(XX)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(XX, proba, label='Model Prediction')
    plt.scatter(X, y, facecolor='gray', edgecolors='none', alpha=0.5, label='Actual Data')
    plt.xlabel('Word Length')
    plt.ylabel('Prediction Accuracy')
    plt.title(f'Effect of Word Length on Prediction Accuracy in {name} Dataset')
    plt.legend()
    plt.show()
