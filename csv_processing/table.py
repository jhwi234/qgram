import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Function to calculate the correct prediction ratio table
def calculate_correct_prediction_ratios(df, min_word_size=3, max_word_size=8):
    # Initialize dictionaries to hold the correct predictions and total counts
    results = {size: [0] * size for size in range(min_word_size, max_word_size + 1)}
    counts = {size: [0] * size for size in range(min_word_size, max_word_size + 1)}
    
    # Process each row in the dataframe
    for index, row in df.iterrows():
        word = row['Original_Word']
        word_size = len(word)
        if word_size < min_word_size or word_size > max_word_size:
            continue
        
        for position in range(word_size):
            if row['Top1_Is_Accurate']:
                results[word_size][position] += 1
            counts[word_size][position] += 1
    
    # Calculate the ratios
    ratios = {size: [round(results[size][i] / counts[size][i], 2) if counts[size][i] > 0 else 0 
                     for i in range(size)] for size in range(min_word_size, max_word_size + 1)}
    
    # Convert to DataFrame for easier display
    ratio_df = pd.DataFrame.from_dict(ratios, orient='index', columns=[f'Position {i+1}' for i in range(max_word_size)])
    return ratio_df

# Function to create and save a visual table as PNG
def save_table_as_png(dataframe, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=dataframe.values,
                     colLabels=dataframe.columns,
                     rowLabels=dataframe.index,
                     cellLoc='center',
                     loc='center')
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=16)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# Function to run the analysis on a given dataset path
def run_analysis(dataset_path, dataset_name):
    df = pd.read_csv(dataset_path)
    ratio_df = calculate_correct_prediction_ratios(df)
    output_png = f'correct_prediction_ratios_{dataset_name}.png'
    save_table_as_png(ratio_df, f'Correct Prediction Ratios for {dataset_name}', output_png)
    return ratio_df

# Define dataset paths
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Run analysis for each dataset
results = {}
for name, path in dataset_paths.items():
    print(f"Running analysis for {name} dataset...")
    results[name] = run_analysis(path, name)
    print(f"Correct prediction ratios for {name} dataset saved to correct_prediction_ratios_{name}.png")
