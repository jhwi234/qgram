import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def calculate_distance_and_filter(df, vowels):
    """
    Filter out rows with missing vowels and calculate the distance to the nearest vowel
    for rows with missing consonants. Returns the filtered and updated DataFrame.
    """
    def is_consonant_missing(tested_word, original_word):
        missing_letter_index = tested_word.find('_')
        if missing_letter_index != -1:
            return original_word[missing_letter_index] not in vowels
        return False
    
    def distance_to_vowel(tested_word, original_word):
        missing_letter_index = tested_word.find('_')
        min_distance = len(original_word)  # Initialize with max possible distance
        for i, letter in enumerate(original_word):
            if letter in vowels:
                distance = abs(i - missing_letter_index)
                min_distance = min(min_distance, distance)
        return min_distance
    
    # Filter DataFrame for rows where the missing letter is a consonant
    filtered_df = df[df.apply(lambda row: is_consonant_missing(row['Tested_Word'], row['Original_Word']), axis=1)].copy()
    
    # Calculate the distance to the nearest vowel for the filtered rows
    # Using .loc to avoid SettingWithCopyWarning
    filtered_df.loc[:, 'Distance_To_Vowel'] = filtered_df.apply(lambda row: distance_to_vowel(row['Tested_Word'], row['Original_Word']), axis=1)
    
    return filtered_df

def preprocess_and_analyze_vowel_distance(path, name):
    """
    Preprocess the dataset and analyze how the distance between the missing consonant and the nearest vowel
    affects the Top1_Is_Accurate metric.
    """
    if not Path(path).exists():
        print(f"The path {path} does not exist.")
        return
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path} due to error: {e}")
        return

    vowels = set('aeèéiîouyæœ')
    df['Top1_Is_Accurate'] = df['Top1_Is_Accurate'].astype(int)
    filtered_df = calculate_distance_and_filter(df, vowels)
    
    if filtered_df.empty:
        print("No data after filtering for missing consonants.")
        return

    overall_mean_accuracy = filtered_df['Top1_Is_Accurate'].mean()
    distance_means = filtered_df.groupby('Distance_To_Vowel')['Top1_Is_Accurate'].mean()

    # Calculate the sample counts for each distance
    sample_counts = filtered_df['Distance_To_Vowel'].value_counts().sort_index()

    print(f"\n{name} Dataset Analysis:")
    print(f"Overall Mean Accuracy: {overall_mean_accuracy:.2f}")
    for distance, mean_accuracy in distance_means.items():
        print(f"Mean Accuracy for Distance {distance}: {mean_accuracy:.2f}")
    
    # Pass both distance_means and sample_counts to the plotting function
    plot_accuracy_vs_distance(distance_means, sample_counts, name)

def plot_accuracy_vs_distance(distance_means, sample_counts, dataset_name):
    distances = sorted(distance_means.index)
    accuracies = distance_means.values
    
    plt.figure(figsize=(12, 6))
    plt.plot(distances, accuracies, marker='o', linestyle='-', color='blue')
    plt.fill_between(distances, accuracies, alpha=0.1, color='blue')
    
    # Annotate the plot with sample counts
    for i, distance in enumerate(distances):
        plt.text(distance, accuracies[i], f' n={sample_counts[distance]}', verticalalignment='bottom')
    
    plt.xlabel('Distance to Nearest Vowel')
    plt.ylabel('Mean Prediction Accuracy')
    plt.title(f'{dataset_name} Dataset: Mean Prediction Accuracy vs. Distance to Nearest Vowel for Missing Consonants')
    plt.grid(True)
    plt.xticks(np.arange(min(distances), max(distances)+1, 1.0))
    plt.show()

def main():
    datasets = {
        "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv',
        "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv',
        "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv',
        "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv',
        "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv'
    }

    for name, path in datasets.items():
        print(f"\nProcessing {name} dataset...")
        preprocess_and_analyze_vowel_distance(path, name)

if __name__ == "__main__":
    main()
