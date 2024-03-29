import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

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

    # Normalize positions into 10 bins
    df['Normalized_Position'] = df['Missing_Letter_Position'].apply(lambda x: min(x, 10))

    # Calculate average accuracy for each normalized position
    accuracy_by_position = df.groupby('Normalized_Position')['Top1_Is_Accurate'].mean().reset_index()

    return accuracy_by_position

def plot_line(data, name):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")
    
    # Make the plot larger and set an aspect ratio
    plt.figure(figsize=(10, 6))
    
    # Use a colorblind-friendly color
    color = sns.color_palette("colorblind")[0]  # Selects the first color from the colorblind-friendly palette
    
    # Create the bar plot with Seaborn for automatic aesthetics improvements, using a colorblind-friendly color
    bars = sns.barplot(x='Normalized_Position', y='Top1_Is_Accurate', data=data, color=color)
    
    # Set more descriptive titles and labels
    bars.set_title(f'Prediction Accuracy by Letter Position in {name} Corpus', fontsize=16)
    bars.set_xlabel('Position of Missing/Incorrect Letter', fontsize=14)
    bars.set_ylabel('Average Prediction Accuracy (%)', fontsize=14)
    
    # Improve tick marks for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adding value labels on top of each bar
    for index, row in data.iterrows():
        bars.text(index, row.Top1_Is_Accurate, round(row.Top1_Is_Accurate, 2), color='black', ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.show()

def main():
    dataset_paths = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

    for name, path in dataset_paths.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        plot_line(data, name)

if __name__ == "__main__":
    main()
