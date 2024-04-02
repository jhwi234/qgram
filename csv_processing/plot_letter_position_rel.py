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
    """
    Plot the accuracy of predictions across different bins of the relative positions of the missing letter.

    Args:
    - data (DataFrame): The preprocessed data containing binned relative positions and their corresponding accuracies.
    - name (str): The name of the dataset being visualized, used for titling the plot.
    """
    # Set the visual style of the plots to be more readable
    sns.set_style("whitegrid")
    
    # Create a figure with a specified size for clarity
    plt.figure(figsize=(10, 6))
    
    # Choose a color from the colorblind-friendly palette
    color = sns.color_palette("colorblind")[0]
    
    # Create a bar plot with binned relative positions on the x-axis and accuracy on the y-axis
    bars = sns.barplot(x='Binned_Position', y='Top1_Is_Accurate', data=data, color=color)
    
    # Set the title and labels of the plot, with increased font sizes for readability
    bars.set_title(f'Prediction Accuracy by Binned Relative Letter Position in {name} Corpus', fontsize=20)
    bars.set_xlabel('Binned Relative Position of Missing Letter', fontsize=18)
    bars.set_ylabel('Mean Prediction Accuracy (%)', fontsize=18)
    
    # Customize tick marks for better readability
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Enhance grid lines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
    
    # Optionally, adjust the tick labels to reflect the bin ranges or categories if needed
    
    # Add percentage labels above each bar for clarity
    for index, row in data.iterrows():
        bars.text(index, row.Top1_Is_Accurate, f'{round(row.Top1_Is_Accurate, 2)}%', color='black', ha="center", va='bottom', fontsize=14)
    
    # Adjust layout to accommodate labels
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to preprocess and plot data from multiple datasets.
    """
    # Define paths to datasets
    dataset_paths = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

    # Iterate over each dataset, preprocess, and plot the data
    for name, path in dataset_paths.items():
        print(f"\nAnalyzing {name} Dataset...")
        data = preprocess_data(path)
        plot_line(data, name)

if __name__ == "__main__":
    main()
