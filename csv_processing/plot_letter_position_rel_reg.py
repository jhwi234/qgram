import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_data(path):
    df = pd.read_csv(path)
    
    def relative_position(tested_word, original_word):
        tested_word, original_word = str(tested_word), str(original_word)
        for i in range(min(len(tested_word), len(original_word))):
            if tested_word[i] != original_word[i]:
                return (i / (len(original_word) - 1)) if len(original_word) > 1 else 0
        return None

    df['Normalized_Relative_Position'] = df.apply(
        lambda row: relative_position(row['Tested_Word'], row['Original_Word']), axis=1
    )

    df['Binned_Position'] = pd.cut(df['Normalized_Relative_Position'], bins=8, labels=False) + 1
    accuracy_by_position = df.groupby('Binned_Position')['Top1_Is_Accurate'].mean().reset_index()

    return accuracy_by_position

def perform_regression_and_plot(accuracy_by_position, title):
    X = accuracy_by_position['Binned_Position'].values.reshape(-1, 1)
    y = accuracy_by_position['Top1_Is_Accurate'].values

    model = LinearRegression()
    model.fit(X, y)

    predicted_accuracies = model.predict(X)
    r_squared = model.score(X, y)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Original Accuracies')
    plt.plot(X, predicted_accuracies, color='red', label='Predicted Accuracies')
    plt.title(f'Regression Analysis of Letter Position Accuracy for {title}')
    plt.xlabel('Binned Position')
    plt.ylabel('Mean Accuracy')
    plt.xticks(np.arange(1, 11, 1))
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"{title} - R-squared: {r_squared}\n")
    print(f"{title} - Coefficients: {model.coef_}\n")
    print(f"{title} - Intercept: {model.intercept_}\n")


def main():
    dataset_paths = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
    }
    
    for title, path in dataset_paths.items():
        accuracy_by_position = preprocess_data(path)
        perform_regression_and_plot(accuracy_by_position, title)

if __name__ == "__main__":
    main()