import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def preprocess_data(path):
    """
    Preprocess data by calculating the relative position of missing letters in words and normalizing.
    The relative position is calculated as the position of the missing letter divided by the total length
    of the original word - 1, allowing comparison across words of different lengths.

    Args:
    - path (Path or str): The path to the CSV file containing the dataset.

    Returns:
    - DataFrame: A DataFrame with columns for relative position and the mean accuracy of predictions for that position.
    """
    # Read the dataset from the given path
    df = pd.read_csv(path)
    
    def relative_position(tested_word, original_word):
        """
        Calculate the relative position of the missing letter in the tested word
        compared to the original word's length. Adjusts for words of different lengths
        by normalizing the position over the length of the word.

        Args:
        - tested_word (str): The word with a missing letter.
        - original_word (str): The original word for comparison.

        Returns:
        - float or None: The relative position of the first differing letter, normalized, or None if no difference is found.
        """
        tested_word, original_word = str(tested_word), str(original_word)
        for i in range(min(len(tested_word), len(original_word))):
            if tested_word[i] != original_word[i]:
                # Normalize the position: 0 for the first letter, 1 for the last letter
                return (i / (len(original_word) - 1)) if len(original_word) > 1 else 0
        return None

    # Apply the function to each row to compute the normalized relative position of the missing letter
    df['Normalized_Relative_Position'] = df.apply(
        lambda row: relative_position(row['Tested_Word'], row['Original_Word']), axis=1
    )

    # Bin the normalized relative positions into 10 bins
    df['Binned_Position'] = pd.cut(df['Normalized_Relative_Position'], bins=10, labels=False) + 1

    # Group by the binned position and calculate the mean accuracy for each bin
    accuracy_by_position = df.groupby('Binned_Position')['Top1_Is_Accurate'].mean().reset_index()

    return accuracy_by_position

def plot_line(data, name):
    # Use Seaborn's theme setting for a clean aesthetic
    sns.set_theme(style="whitegrid")

    # Enhanced figure size for better readability
    plt.figure(figsize=(12, 8))

    # Instead of using a diverse color palette for each bar, use a single color
    # This avoids the issue with the deprecation warning regarding palette use without 'hue'
    color = sns.color_palette("deep", 2)  # This selects a single color from the 'deep' palette

    # Creating a bar plot with a consistent color scheme
    bars = sns.barplot(
        x='Binned_Position',
        y='Top1_Is_Accurate',
        data=data,
        color=color[0]  # Apply the single selected color
    )

    # Customizing the plot's title and axis labels with more readable fonts
    bars.set_title(f'Accuracy by Relative Letter Position - {name}', fontsize=22, fontweight='bold')
    bars.set_xlabel('Relative Position of Missing Letter', fontsize=20)
    bars.set_ylabel('Mean Accuracy (%)', fontsize=20)

    # Adjusting tick parameters for better readability
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Adding percentage labels above each bar for enhanced clarity
    for index, row in data.iterrows():
        bars.text(index, row.Top1_Is_Accurate, f"{round(row.Top1_Is_Accurate, 2)}%", color='black', ha="center", fontsize=14)

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to preprocess and plot data from multiple datasets.
    """
    # Define paths to datasets
    dataset_paths = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
    }

    # Iterate over each dataset, preprocess, and plot the data
    for name, path in dataset_paths.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        plot_line(data, name)

if __name__ == "__main__":
    main()
