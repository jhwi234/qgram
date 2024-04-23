import pandas as pd
import numpy as np
from pygam import LogisticGAM, s
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import logging
from pathlib import Path

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

def load_data(path):
    data = pd.read_csv(path)
    logging.info(f"Columns in {path}: {data.columns}")
    if 'Top1_Is_Accurate' not in data.columns:
        logging.error(f"Required column 'Top1_Is_Accurate' is missing in the file {path}")
        return None, None
    # Calculate normalized index for the first character of each word
    data['Normalized_Index'] = data['Tested_Word'].apply(lambda x: 1/len(x) if len(x) > 0 else 0)
    X = data[['Normalized_Index']]  # Use normalized index as a predictor
    y = data['Top1_Is_Accurate'].astype(int)
    return X, y

def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    predicted_probabilities = model.predict_proba(X)
    accuracy = accuracy_score(y, predictions)
    loss = log_loss(y, predicted_probabilities)
    aic = model.statistics_.get('AIC', np.nan)  # Use np.nan instead of 'Not available'
    bic = model.statistics_.get('BIC', np.nan)  # Use np.nan instead of 'Not available'
    return accuracy, loss, aic, bic

# Fitting models and calculating metrics
results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, path in dataset_paths.items():
    X, y = load_data(path)
    if X is None or y is None:
        continue
    for num_splines in range(5, 41, 5):  # Testing with 5, 10, 15, ..., 40 splines
        gam = LogisticGAM(s(0, n_splines=num_splines)).fit(X, y)
        cv_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            gam.fit(X_train, y_train)
            accuracy, loss, aic, bic = calculate_metrics(gam, X_test, y_test)
            cv_scores.append((accuracy, loss, aic, bic))
        avg_cv_scores = pd.DataFrame(cv_scores, columns=['Accuracy', 'LogLoss', 'AIC', 'BIC']).mean()
        results.append((name, num_splines, *avg_cv_scores))

results_df = pd.DataFrame(results, columns=['Dataset', 'Splines', 'Average Accuracy', 'Average LogLoss', 'Average AIC', 'Average BIC'])
print(results_df)
