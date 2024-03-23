import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_data(path):
    df = pd.read_csv(path)
    
    # Function to calculate the position of the missing letter
    def missing_letter_position(tested_word, original_word):
        tested_word, original_word = str(tested_word), str(original_word)
        for i in range(min(len(tested_word), len(original_word))):
            if tested_word[i] != original_word[i]:
                return i + 1  # Position is 1-indexed
        return None

    # Apply the function to calculate missing letter positions
    df['Missing_Letter_Position'] = df.apply(lambda row: missing_letter_position(row['Tested_Word'], row['Original_Word']), axis=1)

    # Normalize positions into 9 bins
    df['Normalized_Position'] = df['Missing_Letter_Position'].apply(lambda x: min(x, 9))

    # Calculate average accuracy for each normalized position
    accuracy_by_position = df.groupby('Normalized_Position')['Top1_Is_Accurate'].mean().reset_index()

    return accuracy_by_position

def plot_line(data, name):
    fig, ax = plt.subplots()
    bars = ax.bar(data['Normalized_Position'], data['Top1_Is_Accurate'], color='skyblue')
    ax.set_title(f'{name} Corpus - Accuracy by Normalized Letter Position', fontsize=14)
    ax.set_xlabel('Normalized Letter Position', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_xticks(data['Normalized_Position'])
    ax.set_xticklabels(data['Normalized_Position'], rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adding value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
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
        plot_line(data, name)

if __name__ == "__main__":
    main()
