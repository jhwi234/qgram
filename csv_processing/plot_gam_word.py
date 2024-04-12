from pygam import LogisticGAM, s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to your datasets
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def load_data(path):
    try:
        data = pd.read_csv(path)
        logging.info(f"Data loaded successfully from {path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return None

def prepare_data(data):
    if 'Top1_Is_Accurate' not in data.columns:
        logging.error("Required column 'Top1_Is_Accurate' is missing")
        return None
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    data['Normalized_Index'] = data.index / (len(data) - 1)
    return data

def fit_model(X, y, n_splines=25):
    gam = LogisticGAM(s(0, n_splines=n_splines)).fit(X, y)
    logging.info("Model fitting complete")
    return gam

def plot_results(XX, proba, X, y, title):
    plt.figure(figsize=(14, 8))
    # Use a colorblind-friendly color for the plot line
    plt.plot(XX, proba, label='Model Prediction', color='blue', linewidth=2)
    # Adjust scatter plot to be colorblind-friendly and more distinct
    plt.scatter(X, y, facecolor='orange', edgecolors='black', alpha=0.7, label='Actual Data', marker='^', s=50)
    plt.xlabel('Normalized Index Position', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)  # Add grid
    plt.tight_layout()  # Adjust layout to make room for label
    plt.show()

def process_dataset(name, path):
    data = load_data(path)
    if data is not None:
        data = prepare_data(data)
        if data is not None:
            X = data[['Normalized_Index']]
            y = data['Top1_Is_Accurate']
            gam = fit_model(X, y)
            XX = np.linspace(X.min(), X.max(), 500)
            proba = gam.predict_proba(XX)
            plot_results(XX, proba, X, y, f'Effect of Normalized Index Position on Prediction Accuracy in {name} Dataset')

for name, path in dataset_paths.items():
    process_dataset(name, path)
