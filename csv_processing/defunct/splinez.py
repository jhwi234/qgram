import pandas as pd
import numpy as np
from pygam import LogisticGAM, s
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Paths to datasets
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
        logging.info(f"Loaded {path} with columns {data.columns}")
        if 'Top1_Is_Accurate' not in data.columns or 'Tested_Word' not in data.columns:
            raise ValueError("Required columns are missing")
        data['Normalized_Index'] = data['Tested_Word'].apply(lambda x: 1 / len(x) if len(x) > 0 else 0)
        X = data[['Normalized_Index']]
        y = data['Top1_Is_Accurate'].astype(int)
        return X, y
    except Exception as e:
        logging.error(f"Error loading data from {path}: {e}")
        return None, None

def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    predicted_probabilities = model.predict_proba(X)
    return accuracy_score(y, predictions), log_loss(y, predicted_probabilities), model.statistics_['AIC']

def perform_cross_validation(X, y, num_splines):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        gam = LogisticGAM(s(0, n_splines=num_splines))
        gam.fit(X_train, y_train)
        scores = calculate_metrics(gam, X_test, y_test)
        cv_scores.append(scores)
    return pd.DataFrame(cv_scores, columns=['Accuracy', 'LogLoss', 'AIC']).mean()

def main():
    results = []
    for name, path in dataset_paths.items():
        X, y = load_data(path)
        if X is not None and y is not None:
            for num_splines in range(10, 31):  # Testing spline counts more granularly
                avg_cv_scores = perform_cross_validation(X, y, num_splines)
                results.append((name, num_splines, *avg_cv_scores))
                logging.info(f"Processed {name} with {num_splines} splines: {avg_cv_scores}")
    results_df = pd.DataFrame(results, columns=['Dataset', 'Splines', 'Average Accuracy', 'Average LogLoss', 'Average AIC'])
    print(results_df)

if __name__ == "__main__":
    main()
