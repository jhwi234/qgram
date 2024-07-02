import pandas as pd
from pathlib import Path
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

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

# Function to preprocess data
def preprocess_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if not {'Tested_Word', 'Top1_Is_Accurate'}.issubset(data.columns):
        raise ValueError("Required columns are missing from the dataset")
    
    # Calculate word length and normalized missing letter position
    data['Word_Length'] = data['Tested_Word'].str.len()
    data['Normalized_Missing_Letter_Position'] = data['Tested_Word'].apply(
        lambda word: word.index('_') / (len(word) - 1) if '_' in word else np.nan
    )
    
    return data

# Function to perform exploratory data analysis
def exploratory_data_analysis(data: pd.DataFrame, dataset_name: str):
    logging.info(f"Exploratory Data Analysis for {dataset_name} dataset")
    
    # Summary statistics
    logging.info(f"Summary statistics:\n{data.describe()}")
    
    # Distribution plots
    sns.histplot(data['Normalized_Missing_Letter_Position'].dropna(), bins=20, kde=True)
    plt.title(f'Distribution of Normalized Missing Letter Position - {dataset_name}')
    plt.show()
    
    sns.countplot(x='Top1_Is_Accurate', data=data)
    plt.title(f'Count of Top1 Is Accurate - {dataset_name}')
    plt.show()

# Function to perform logistic regression analysis
def logistic_regression_analysis(file_path: Path, dataset_name: str):
    data = preprocess_data(file_path)
    
    # Ensure 'Top1_Is_Accurate' is boolean and convert to integer (1 for True, 0 for False)
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    
    # Drop rows with NaN values in 'Normalized_Missing_Letter_Position'
    data = data.dropna(subset=['Normalized_Missing_Letter_Position'])
    
    # Exploratory Data Analysis
    exploratory_data_analysis(data, dataset_name)
    
    # Train-test split
    X = data[['Normalized_Missing_Letter_Position']]
    y = data['Top1_Is_Accurate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handling class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Fit logistic regression model using scikit-learn
    model = LogisticRegression()
    model.fit(X_train_res, y_train_res)
    
    logging.info(f"\n{dataset_name} Dataset Analysis (Logistic Regression):")
    logging.info(classification_report(y_test, model.predict(X_test)))
    
    # Model diagnostics
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    logging.info(f"AUC Score (Logistic Regression): {auc_score:.2f}")
    
    # Cross-validation
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    logging.info(f"Cross-validation AUC scores (Logistic Regression): {cross_val_scores}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_pred_prob, 'o', label='Predicted Probability')
    plt.xlabel('Normalized Missing Letter Position')
    plt.ylabel('Predicted Probability')
    plt.title(f'Logistic Regression Predicted Probability - {dataset_name}')
    plt.legend()
    plt.show()

# Function to perform random forest analysis
def random_forest_analysis(file_path: Path, dataset_name: str):
    data = preprocess_data(file_path)
    
    # Ensure 'Top1_Is_Accurate' is boolean and convert to integer (1 for True, 0 for False)
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    
    # Drop rows with NaN values in 'Normalized_Missing_Letter_Position'
    data = data.dropna(subset=['Normalized_Missing_Letter_Position'])
    
    # Exploratory Data Analysis
    exploratory_data_analysis(data, dataset_name)
    
    # Train-test split
    X = data[['Normalized_Missing_Letter_Position']]
    y = data['Top1_Is_Accurate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handling class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Fit Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)
    
    logging.info(f"\n{dataset_name} Dataset Analysis (Random Forest):")
    logging.info(classification_report(y_test, model.predict(X_test)))
    
    # Model diagnostics
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    logging.info(f"AUC Score (Random Forest): {auc_score:.2f}")
    
    # Cross-validation
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    logging.info(f"Cross-validation AUC scores (Random Forest): {cross_val_scores}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_pred_prob, 'o', label='Predicted Probability')
    plt.xlabel('Normalized Missing Letter Position')
    plt.ylabel('Predicted Probability')
    plt.title(f'Random Forest Predicted Probability - {dataset_name}')
    plt.legend()
    plt.show()

# Perform analysis for each dataset
for dataset_name, file_path in dataset_paths.items():
    logistic_regression_analysis(file_path, dataset_name)
    random_forest_analysis(file_path, dataset_name)
