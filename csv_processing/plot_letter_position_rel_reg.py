from sklearn.linear_model import LinearRegression
import pandas as pd
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
    
    return df[['Normalized_Relative_Position', 'Top1_Is_Accurate']]

def run_regression(data):
    X = data[['Normalized_Relative_Position']]
    y = data['Top1_Is_Accurate']
    reg = LinearRegression().fit(X, y)

    # Return the model's parameters and score
    return {
        'coefficients': reg.coef_,
        'intercept': reg.intercept_,
        'score': reg.score(X, y)  # R^2 score of the model
    }

def main():
    dataset_paths = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
    }

    for name, path in dataset_paths.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        regression_results = run_regression(data)
        print(f"Results for {name}:\n"
              f"Coefficients: {regression_results['coefficients']}\n"
              f"Intercept: {regression_results['intercept']}\n"
              f"R^2 Score: {regression_results['score']}\n")

if __name__ == "__main__":
    main()
