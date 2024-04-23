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
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None

def prepare_data(data):
    required_columns = {'Top1_Is_Accurate'}
    if not required_columns.issubset(data.columns):
        logging.error("Required column 'Top1_Is_Accurate' is missing")
        return None
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    return data

def fit_model(X, y, n_splines=25): # Default number of splines is set to 25
    try:
        gam = LogisticGAM(s(0, n_splines=n_splines)).fit(X, y) # Fit the model with the specified number of splines
        logging.info("Model fitting complete")
        return gam
    except Exception as e:
        logging.error(f"Error fitting model: {str(e)}")
        return None

def adjust_y_axis(proba):
    """Adjust the y-axis of the plot based on the median of model predictions."""
    center_point = np.median(proba)  # Using the median of predictions as the center point
    margin = 0.15  # Margin of 15% above and below the center point
    plt.ylim([max(0, center_point - margin), min(1, center_point + margin)])

def plot_results(XX, proba, X, y, title, config, output_path):
    plt.figure(figsize=config.get('figsize', (14, 8)))
    plt.plot(XX, proba, label='Model Prediction', color=config.get('prediction_color', 'blue'), linewidth=2)
    plt.scatter(X, y, color=config.get('data_color', 'black'), alpha=0.7, label='Actual Data')
    plt.xlabel('Normalized Index Position', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True)
    ticks = np.arange(0.1, 1.1, 0.1)
    plt.xticks(ticks, labels=[f"{tick:.1f}" for tick in ticks])
    if config.get('dynamic_range', True):
        adjust_y_axis(proba)
    plt.tight_layout()
    plt.savefig(output_path)  # Save the figure to the specified path
    plt.close()  # Close the figure to free up memory

def process_dataset(name, path, config):
    output_dir = Path('output/gams')
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    data = load_data(path)
    if data is not None:
        prepared_data = prepare_data(data) # Prepare the data for modeling
        if prepared_data is not None:
            X = prepared_data.index.to_frame(index=False) / len(prepared_data) # Normalize the index
            y = prepared_data['Top1_Is_Accurate'] # Use the 'Top1_Is_Accurate' column as the target variable
            gam = fit_model(X, y) # Fit the GAM model
            if gam:
                XX = np.linspace(0, 1, 500)[:, None] # Generate 500 points between 0 and 1
                proba = gam.predict_proba(XX) # Predict the probabilities for the generated points
                output_path = output_dir / f"{name}_GAM.png"  # Define the path for saving the plot
                plot_results(XX.ravel(), proba, X.to_numpy().ravel(), y, f'Effect of Normalized Index Position on Prediction Accuracy in {name}', config, output_path)

default_plot_config = {
    'figsize': (14, 8),
    'style': 'seaborn-darkgrid',
    'prediction_color': 'blue',
    'data_color': 'black',
    'dynamic_range': True
}

for name, path in dataset_paths.items():
    process_dataset(name, path, default_plot_config)
