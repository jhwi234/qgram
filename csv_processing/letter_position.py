import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Missing_Letter_Position'] = data['Tested_Word'].apply(lambda x: x.find('_') if isinstance(x, str) else -1)
    data['Word_Length'] = data['Original_Word'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    data['Normalized_Missing_Letter_Position'] = data.apply(lambda row: row['Missing_Letter_Position'] / (row['Word_Length'] - 1) if row['Word_Length'] > 1 else 0, axis=1)
    data['Normalized_Position_Bin'] = pd.cut(data['Normalized_Missing_Letter_Position'], bins=10, labels=range(10))
    return data

def logistic_regression_analysis(data):
    data = add_additional_features(data)
    X = data[['Normalized_Missing_Letter_Position', 'Word_Complexity']]
    y = data['Top1_Is_Accurate']
    
    # Standardize features since you're directly using scikit-learn's Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Printing evaluation metrics
    print_evaluation_metrics(y_test, predictions)
    
    # Now call the cross-validation function with the standardized features and target
    logistic_regression_with_cross_validation(X_scaled, y)
    
    return data.groupby('Normalized_Position_Bin', observed=True)['Top1_Is_Accurate'].mean().reset_index()

def add_additional_features(data):
    # Adjusting the function to handle non-string (NaN or float) values in 'Original_Word'
    data['Word_Complexity'] = data['Original_Word'].apply(lambda x: len(set(x)) / len(x) if isinstance(x, str) and len(x) > 0 else 0)
    return data

def logistic_regression_with_cross_validation(X, y):
    
    # It's good practice to scale features when using logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize the LogisticRegression model
    model = LogisticRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean()}")

def print_evaluation_metrics(y_test, predictions_binary):
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions_binary))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions_binary, zero_division=0))

def print_model_diagnostics(model):
    print("\nPseudo R-squared: {:.4f}".format(model.prsquared))
    print("Log-Likelihood: {:.4f}".format(model.llf))
    print("AIC: {:.4f}".format(model.aic))
    print("BIC: {:.4f}".format(model.bic))

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        accuracy = logistic_regression_analysis(data)
        print(f"\n{name} Dataset Normalized Position Accuracy:\n", accuracy)

if __name__ == "__main__":
    main()
