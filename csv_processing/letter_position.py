import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
    X = sm.add_constant(data[['Normalized_Missing_Letter_Position']])
    y = data['Top1_Is_Accurate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = sm.Logit(y_train, X_train).fit(disp=0)
    predictions = model.predict(X_test)
    predictions_binary = [1 if x > 0.5 else 0 for x in predictions]
    
    print_evaluation_metrics(y_test, predictions_binary)
    print_model_diagnostics(model)
    plot_logistic_regression(data, X_test, predictions)
    
    return data.groupby('Normalized_Position_Bin', observed=True)['Top1_Is_Accurate'].mean().reset_index()

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

def plot_logistic_regression(data, X_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Normalized_Missing_Letter_Position'], data['Top1_Is_Accurate'], color='red', label='Actual data')
    plt.plot(X_test['Normalized_Missing_Letter_Position'], predictions, color='blue', label='Logistic Regression Curve')
    plt.xlabel('Normalized Missing Letter Position')
    plt.ylabel('Prediction Accuracy')
    plt.title('Logistic Regression Analysis')
    plt.legend()
    plt.show()

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

    for name, path in datasets.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        accuracy = logistic_regression_analysis(data)
        print(f"\n{name} Dataset Normalized Position Accuracy:\n", accuracy)
if __name__ == "__main__":
    main()
