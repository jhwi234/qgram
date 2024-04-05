from pathlib import Path
import pandas as pd

def extract_neighbors(tested_word):
    missing_letter_index = tested_word.find('_')
    before = tested_word[missing_letter_index - 1] if missing_letter_index > 0 else None
    after = tested_word[missing_letter_index + 1] if missing_letter_index < len(tested_word) - 1 else None
    return before, after

def analyze_dataset(file_path):
    data = pd.read_csv(file_path)
    
    # Clean the data if necessary
    data.dropna(subset=['Tested_Word', 'Original_Word'], inplace=True)
    
    # Extract neighboring letters
    data['Before_Letter'], data['After_Letter'] = zip(*data['Tested_Word'].apply(extract_neighbors))
    
    # Group and calculate accuracy for before and after letters separately
    accuracy_by_before = data.groupby('Before_Letter')['Top1_Is_Accurate'].mean().sort_values(ascending=False).head(10)
    accuracy_by_after = data.groupby('After_Letter')['Top1_Is_Accurate'].mean().sort_values(ascending=False).head(10)
    
    return accuracy_by_before, accuracy_by_after

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
    }
    
    for dataset_name, file_path in datasets.items():
        print(f"Analyzing {dataset_name} Dataset at {file_path}:")
        accuracy_by_before, accuracy_by_after = analyze_dataset(file_path)
        print("Top 10 Most Predictive Letters Before the Missing Letter and their Predictive Accuracy:")
        print(accuracy_by_before.to_string())
        print("\nTop 10 Most Predictive Letters After the Missing Letter and their Predictive Accuracy:")
        print(accuracy_by_after.to_string())
        print("\n---\n")

if __name__ == "__main__":
    main()
