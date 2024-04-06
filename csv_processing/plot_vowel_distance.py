import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def preprocess_and_analyze_vowel_distance(path, name):
    """
    Enhanced analysis to include statistical summaries alongside plots.
    """
    df = pd.read_csv(path)
    vowels = set('aeèéiîouyæœ')

    def distance_to_vowel(tested_word, original_word):
        # Find the distance of the missing letter from the nearest vowel.
        min_distance = len(original_word)  # Initialize with max possible distance
        for i, letter in enumerate(original_word):
            if letter in vowels:
                distance = abs(i - tested_word.find('_'))
                min_distance = min(min_distance, distance)
        return min_distance

    df['Distance_To_Vowel'] = df.apply(lambda row: distance_to_vowel(row['Tested_Word'], row['Original_Word']), axis=1)
    df['Top1_Is_Accurate'] = df['Top1_Is_Accurate'].astype(int)  # Ensure accuracy is in binary format

    # Calculate mean accuracy for different distance ranges
    overall_mean_accuracy = df['Top1_Is_Accurate'].mean()
    distance_means = df.groupby('Distance_To_Vowel')['Top1_Is_Accurate'].mean()

    # Terminal output for mean accuracies
    print(f"\n{name} Dataset Analysis:")
    print(f"Overall Mean Accuracy: {overall_mean_accuracy:.2f}")
    for distance, mean_accuracy in distance_means.items():
        print(f"Mean Accuracy for Distance {distance}: {mean_accuracy:.2f}")

    # Plotting
    distances = sorted(df['Distance_To_Vowel'].unique())
    accuracies = [distance_means.get(distance, 0) for distance in distances]

    plt.figure(figsize=(12, 6))
    plt.plot(distances, accuracies, marker='o', linestyle='-', color='blue')
    plt.fill_between(distances, accuracies, alpha=0.1, color='blue')
    plt.xlabel('Distance to Nearest Vowel')
    plt.ylabel('Mean Prediction Accuracy')
    plt.title(f'{name} Dataset: Mean Prediction Accuracy vs. Distance to Nearest Vowel')
    plt.grid(True)
    plt.xticks(np.arange(min(distances), max(distances)+1, 1.0))
    plt.show()

def main():
    datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
    }

    for name, path in datasets.items():
        print(f"\nProcessing {name} dataset...")
        preprocess_and_analyze_vowel_distance(path, name)

if __name__ == "__main__":
    main()
