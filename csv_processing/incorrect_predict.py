import pandas as pd
from pathlib import Path

# Improved version with comments on changes
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

sonority_hierarchy = {
    # Voiceless stops are mapped to level 1 and marked as voiceless (0).
    **{letter: (1, 0) for letter in "PTKCQX"},  
    # Voiced stops are at the same level but marked as voiced (1).
    **{letter: (1, 1) for letter in "BDG"},   
    # Voiceless fricatives are more sonorous than stops, placed at level 2 and marked as voiceless.
    **{letter: (2, 0) for letter in "FSH"},  
    # Voiced fricatives are at the same level as voiceless fricatives but are voiced.
    **{letter: (2, 1) for letter in "VZJ"},   
    # Nasals come next as they're more sonorous than fricatives but less than liquids and glides.
    **{letter: (3, 1) for letter in "MN"},       
    # Liquids and glides are grouped together, reflecting their closer sonority to vowels. 
    **{letter: (4, 1) for letter in "LRW"},       
    # Vowels, being the most sonorous, are placed at the top, including uppercase "Æ" and "Œ".
    **{letter: (5, 1) for letter in "ÆŒAEIOUY"}     
}

def calculate_sonority_distance(correct, predicted):
    """
    Calculate the refined distance based on sonority hierarchy between the correct and predicted letters.
    This includes adjustments for differences within the same category and voicing.

    Parameters:
    - correct (str): The correct letter.
    - predicted (str): The predicted letter.

    Returns:
    - float: The refined sonority distance considering level, voicing, and category.
    """
    correct_sonority, predicted_sonority = sonority_hierarchy.get(correct.upper(), (0, 0)), sonority_hierarchy.get(predicted.upper(), (0, 0))
    level_difference = abs(correct_sonority[0] - predicted_sonority[0])
    # Adjust for voicing differences
    voicing_difference = 0.5 if (correct_sonority[0] == predicted_sonority[0] and correct_sonority[1] != predicted_sonority[1]) else 0
    # New rule: within the same category but not the same letter, distance is 0.25
    within_category_difference = 0.25 if (correct_sonority[0] == predicted_sonority[0] and correct.upper() != predicted.upper()) else 0

    # The total distance considers both voicing and within-category differences
    total_distance = max(voicing_difference, within_category_difference) if level_difference == 0 else level_difference
    return total_distance

def analyze_mispredictions(dataset_path: Path) -> pd.DataFrame:
    """
    Analyzes mispredictions in the given dataset, providing the most frequent mistakes, their frequencies,
    and the refined calculation of the sonority distance considering new rules.
    
    Parameters:
    - dataset_path (Path): Path to the dataset file.
    
    Returns:
    - pd.DataFrame: DataFrame with the most frequently mistaken letters, the frequency of each mistake,
                    and the refined sonority distance.
    """
    try:
        # Load the dataset
        dataset_df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return pd.DataFrame()

    # Filter out correct predictions to focus on mispredictions
    mispredicted_df = dataset_df[dataset_df['Correct_Letter'] != dataset_df['Top1_Predicted_Letter']]
    
    # Analyze the frequency of mistaken letters for each correct letter
    misprediction_analysis = mispredicted_df.groupby(['Correct_Letter', 'Top1_Predicted_Letter']).size().unstack(fill_value=0)
    
    # Find the most frequently mistaken letter for each correct letter and its frequency
    most_frequent_mistakes = misprediction_analysis.idxmax(axis=1)
    mistake_frequencies = misprediction_analysis.max(axis=1)
    
    # Use the refined sonority distance calculation
    sonority_distances = most_frequent_mistakes.index.to_series().apply(lambda x: calculate_sonority_distance(x, most_frequent_mistakes[x]))
    
    # Combine into a DataFrame for detailed analysis
    detailed_analysis = pd.DataFrame({
        'Most_Frequent_Mistake': most_frequent_mistakes,
        'Frequency': mistake_frequencies,
        'Sonority_Distance': sonority_distances
    })
    
    return detailed_analysis

# Improved output function (commented out to avoid execution)
for dataset_name, dataset_path in dataset_paths.items():
    print(f"Analyzing dataset: {dataset_name}")
    detailed_analysis = analyze_mispredictions(dataset_path)
    print(detailed_analysis)
    print("\n" + "="*50 + "\n")
