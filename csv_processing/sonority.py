import pandas as pd

# Sonority Hierarchy Dictionary with case insensitivity and voicing distinctions.
# This dictionary maps letters to a tuple consisting of their sonority level and voicing status.
# Sonority levels range from 1 to 5, with 1 being the least sonorous (e.g., stops and fricatives)
# and 5 being the most sonorous (vowels). Voicing is indicated with a binary flag, where 0
# represents voiceless sounds and 1 represents voiced sounds.
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
    # This group includes both liquids (L, R) and the glide (W). Glides are often considered 
    # to be close to vowels in terms of sonority, hence their placement here.
    # Note: Including 'Y' as it functions as both a vowel and a glide depending on context.
    **{letter: (4, 1) for letter in "LRW"},       
    # Vowels, being the most sonorous, are placed at the top.
    # Including both the main vowels and 'Y' when it functions as a vowel.
    **{letter: (5, 1) for letter in "AEIOUY"}     
}

# Now, expand to include both uppercase and lowercase for case insensitivity.
sonority_hierarchy = {**{letter.upper(): value for letter, value in sonority_hierarchy.items()},
                      **{letter.lower(): value for letter, value in sonority_hierarchy.items()}}

# Function to calculate the modified sonority distance between two letters.
# This distance is a measure of how phonetically different two letters are,
# considering both their sonority levels and voicing status.
def sonority_distance(correct_letter, predicted_letter):
    # If both letters are the same, their distance is 0 (exact match).
    if correct_letter == predicted_letter:
        return 0  
    
    # Retrieve the sonority level and voicing status for both letters.
    correct_val = sonority_hierarchy.get(correct_letter, (0, 0))
    predicted_val = sonority_hierarchy.get(predicted_letter, (0, 0))
    
    # Calculate the absolute difference in sonority levels between the two letters.
    sonority_distance = abs(correct_val[0] - predicted_val[0])
    # Calculate the difference in voicing status, scaled by 0.5 (since voicing is a "half step").
    voicing_distance = abs(correct_val[1] - predicted_val[1]) * 0.5
    
    # The total distance is the sum of the sonority and voicing differences.
    return sonority_distance + voicing_distance

# Paths to various datasets containing letter predictions and their correct counterparts.
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Loop through each dataset to calculate and report the average sonority distance
# for incorrectly predicted letters.
for dataset_name, file_path in datasets.items():
    # Load the dataset from the given file path.
    data = pd.read_csv(file_path)

    # Apply the modified sonority distance calculation for each row in the dataset,
    # comparing the correct letter to the top predicted letter.
    data['Refined_Sonority_Distance'] = data.apply(
        lambda row: sonority_distance(row['Correct_Letter'], row['Top1_Predicted_Letter']), axis=1
    )

    # Filter the dataset to include only rows where the prediction was incorrect
    # (distance != 0) and calculate the average distance for these incorrect predictions.
    average_distance_incorrect = data.loc[data['Refined_Sonority_Distance'] != 0, 'Refined_Sonority_Distance'].mean()
    
    # Print out the average distance, providing an insight into the typical magnitude
    # of phonetic prediction errors within each dataset.
    print(f"Average Sonority Distance for Incorrect Predictions in {dataset_name}: {average_distance_incorrect}")
