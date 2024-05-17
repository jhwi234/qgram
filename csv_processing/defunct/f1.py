import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from pathlib import Path

def calculate_metrics_and_save(data_path, output_csv):
    # Ensure the output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the data
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully from {data_path}")
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return
    
    # Prepare the label binarizer with the unique letters found in this specific dataset
    unique_letters = pd.concat([data['Correct_Letter'], data['Top1_Predicted_Letter']]).unique()
    lb = LabelBinarizer()
    lb.fit(unique_letters)
    
    # Binarize the correct and predicted labels
    true_labels = lb.transform(data['Correct_Letter'])
    predicted_labels = lb.transform(data['Top1_Predicted_Letter'])
    
    # Calculate precision, recall, and F1 score for each letter found in this dataset
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, labels=np.arange(len(lb.classes_)))
    
    # Create a DataFrame to display the metrics per letter
    letter_metrics = pd.DataFrame({
        'Letter': lb.classes_,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })
    
    # Save the metrics to a CSV file
    letter_metrics.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")

# Directory for output files using pathlib
output_dir = Path('data/outputs/csv')

# Define dataset paths (assuming they are located in a specific directory)
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Process each dataset
for key, data_path in dataset_paths.items():
    output_csv = output_dir / f'{key}_metrics.csv'
    print(f"Processing {data_path}")  # Debug: Print the file being processed
    calculate_metrics_and_save(data_path, output_csv)
