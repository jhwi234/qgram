import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pygam import LogisticGAM, s

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
    """Load dataset from specified filepath."""
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None

def prepare_data(data):
    """Prepare data by ensuring necessary columns exist and indexing is normalized."""
    required_columns = ['Top1_Is_Accurate']
    if not all(column in data.columns for column in required_columns):
        logging.error("Required column 'Top1_Is_Accurate' is missing")
        return None
    try:
        data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
        data['Normalized_Index'] = data.index / (len(data))
        return data
    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        return None

def fit_model(X, y, n_splines=25):
    """Fit a logistic GAM to the data."""
    try:
        gam = LogisticGAM(s(0, n_splines=n_splines)).fit(X, y)
        logging.info("Model fitting complete")
        return gam
    except Exception as e:
        logging.error(f"Error fitting model: {str(e)}")
        return None

def plot_results(XX, proba, X, y, title, config):
    """Generate plots of model predictions and actual data."""
    plt.figure(figsize=config.get('figsize', (14, 8)))
    plt.plot(XX, proba, label='Model Prediction', color=config.get('prediction_color', 'blue'), linewidth=2)
    plt.scatter(X, y, color=config.get('data_color', 'black'), alpha=0.7, label='Actual Data')
    plt.xlabel('Normalized Index Position', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True)
    if config.get('dynamic_range', True):
        adjust_y_axis(y)
    plt.tight_layout()
    plt.show()

def adjust_y_axis(y):
    """Adjust the y-axis of the plot based on data percentiles."""
    y_min, y_max = np.percentile(y, [10, 90])
    y_range = y_max - y_min
    plt.ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

def process_dataset(name, path, config):
    """Process each dataset: load, prepare, model, and plot results."""
    data = load_data(path)
    if data is not None and not data.empty:  # Check if data is not None and not empty
        prepared_data = prepare_data(data)
        if prepared_data is not None:
            X = prepared_data[['Normalized_Index']]
            y = prepared_data['Top1_Is_Accurate']
            gam = fit_model(X, y)
            if gam:
                XX = np.linspace(X['Normalized_Index'].min(), X['Normalized_Index'].max(), 500)
                proba = gam.predict_proba(XX)
                plot_results(XX.ravel(), proba, X['Normalized_Index'].to_numpy(), y, f'Effect of Normalized Index Position on Prediction Accuracy in {name}', config)

default_plot_config = {
    'figsize': (14, 8),
    'style': 'seaborn-darkgrid',
    'prediction_color': 'blue',
    'data_color': 'black',
    'dynamic_range': True
}

for name, path in dataset_paths.items():
    process_dataset(name, path, default_plot_config)
