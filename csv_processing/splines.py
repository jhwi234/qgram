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
    logging.info(f"Loaded {path} with columns {data.columns}")
    if 'Top1_Is_Accurate' not in data.columns:
        logging.error(f"Required column 'Top1_Is_Accurate' is missing in {path}")
        return None, None
    data['Normalized_Index'] = data['Tested_Word'].apply(lambda x: 1 / len(x) if len(x) > 0 else 0)
    X = data[['Normalized_Index']]
    y = data['Top1_Is_Accurate'].astype(int)
    return X, y

def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    predicted_probabilities = model.predict_proba(X)
    return accuracy_score(y, predictions), log_loss(y, predicted_probabilities), model.statistics_['AIC']

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, path in dataset_paths.items():
    X, y = load_data(path)
    if X is None or y is None:
        continue
    for num_splines in range(5, 41):  # More granular spline count
        gam = LogisticGAM(s(0, n_splines=num_splines))
        cv_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            gam.fit(X_train, y_train)
            accuracy, loss, aic = calculate_metrics(gam, X_test, y_test)
            cv_scores.append((accuracy, loss, aic))
        avg_cv_scores = pd.DataFrame(cv_scores, columns=['Accuracy', 'LogLoss', 'AIC']).mean()
        results.append((name, num_splines, avg_cv_scores['Accuracy'], avg_cv_scores['LogLoss'], avg_cv_scores['AIC']))

results_df = pd.DataFrame(results, columns=['Dataset', 'Splines', 'Average Accuracy', 'Average LogLoss', 'Average AIC'])
print(results_df)
