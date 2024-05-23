import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pygam import LogisticGAM, s
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to datasets
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None

def prepare_data(data):
    """
    Prepare data by calculating word length and normalized missing letter index.
    """
    required_columns = {'Top1_Is_Accurate', 'Tested_Word'}
    if not required_columns.issubset(data.columns):
        logging.error("Required columns are missing")
        return None

    # Calculate lengths of each word and the normalized index of the missing letter
    data['Word_Length'] = data['Tested_Word'].str.len()
    data['Normalized_Missing_Index'] = data['Tested_Word'].apply(
        lambda word: word.index('_') / (len(word) - 1) if '_' in word else np.nan
    )
    
    return data

def fit_model(X, y, n_splines=15):
    """
    Fit a logistic GAM model.
    """
    try:
        gam = LogisticGAM(s(0, n_splines=n_splines)).fit(X, y)
        logging.info("Model fitting complete")
        return gam
    except Exception as e:
        logging.error(f"Error fitting model: {str(e)}")
        return None

def plot_combined_dataset(XX, proba, X, y, config, output_path):
    """
    Plot the results of the combined dataset on a single graph.
    """
    plt.figure(figsize=config.get('figsize', (14, 8)))
    
    plt.plot(XX, proba, label='Model Prediction', linewidth=2)
    plt.scatter(X, y, alpha=0.7, label='Actual Data')
    
    plt.xlabel('Normalized Missing Index', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title('Effect of Normalized Missing Index on Prediction Accuracy for Combined Dataset', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1), labels=[f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_combined_dataset(dataset_paths, config):
    """
    Process all datasets, combine them, fit a single model, and plot the results.
    """
    combined_data = pd.DataFrame()
    
    for path in dataset_paths.values():
        data = load_data(path)
        if data is not None:
            prepared_data = prepare_data(data)
            if prepared_data is not None:
                combined_data = pd.concat([combined_data, prepared_data], ignore_index=True)
    
    if not combined_data.empty:
        X = combined_data[['Normalized_Missing_Index']].dropna()
        y = combined_data.loc[X.index, 'Top1_Is_Accurate']
        gam = fit_model(X, y)
        if gam:
            XX = np.linspace(0, 1, 1000)[:, None]
            proba = gam.predict_proba(XX)
            output_path = Path('output/gams/combined_dataset_GAM_df.png')
            plot_combined_dataset(XX.ravel(), proba, X.to_numpy().ravel(), y, config, output_path)

default_plot_config = {
    'figsize': (14, 8),
    'style': 'seaborn-darkgrid',
    'prediction_color': 'blue',
    'data_color': 'black',
    'dynamic_range': True
}

process_combined_dataset(dataset_paths, default_plot_config)
