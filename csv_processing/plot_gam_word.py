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

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return pd.DataFrame()

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data by calculating word length and normalized missing letter index.
    """
    required_columns = {'Top1_Is_Accurate', 'Tested_Word'}
    if not required_columns.issubset(data.columns):
        logging.error("Required columns are missing")
        return pd.DataFrame()

    # Calculate lengths of each word and the normalized index of the missing letter
    data['Word_Length'] = data['Tested_Word'].str.len()
    data['Normalized_Missing_Index'] = data['Tested_Word'].str.find('_') / (data['Word_Length'] - 1)
    data.replace({'Normalized_Missing_Index': {np.inf: np.nan, -np.inf: np.nan}}, inplace=True)
    
    return data

def fit_model(X: pd.DataFrame, y: pd.Series, n_splines: int = 15) -> LogisticGAM:
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

def adjust_y_axis(proba: np.ndarray):
    """
    Adjust the y-axis based on the median of the predicted probabilities.
    """
    center_point = np.median(proba)
    margin = 0.30
    plt.ylim([max(0, center_point - margin), min(1, center_point + margin)])

def plot_results(XX: np.ndarray, proba: np.ndarray, X: np.ndarray, y: np.ndarray, title: str, config: dict, output_path: Path):
    """
    Plot the results of the GAM model predictions against the actual data.
    """
    plt.figure(figsize=config.get('figsize', (14, 8)))
    plt.plot(XX, proba, label='Model Prediction', color=config.get('prediction_color', 'blue'), linewidth=2)
    plt.scatter(X, y, color=config.get('data_color', 'black'), alpha=0.7, label='Actual Data')
    plt.xlabel('Normalized Missing Index', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1), labels=[f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)])
    if config.get('dynamic_range', True):
        adjust_y_axis(proba)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def process_dataset(name: str, path: Path, config: dict):
    """
    Process each dataset: load data, prepare it, fit the model, and plot results.
    """
    data = load_data(path)
    if not data.empty:
        prepared_data = prepare_data(data)
        if not prepared_data.empty:
            X = prepared_data[['Normalized_Missing_Index']].dropna()
            y = prepared_data.loc[X.index, 'Top1_Is_Accurate']
            gam = fit_model(X, y)
            if gam:
                XX = np.linspace(0, 1, 1000)[:, None]
                proba = gam.predict_proba(XX)
                output_path = Path('output/gams') / f"{name}_GAM_df.png"
                plot_results(XX.ravel(), proba, X.to_numpy().ravel(), y, f'Effect of Normalized Missing Index on Prediction Accuracy in {name}', config, output_path)

default_plot_config = {
    'figsize': (14, 8),
    'style': 'seaborn-darkgrid',
    'prediction_color': 'blue',
    'data_color': 'black',
    'dynamic_range': True
}

for name, path in dataset_paths.items():
    process_dataset(name, path, default_plot_config)
