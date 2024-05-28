import pandas as pd

# Define dataset paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Function to process each dataset and return a DataFrame with results
def process_dataset(name, path):
    data = pd.read_csv(path)
    predicted_letter_counts = data['Top1_Predicted_Letter'].value_counts().sort_index()
    accurate_predicted_letter_counts = data[data['Top1_Is_Accurate']]['Top1_Predicted_Letter'].value_counts().sort_index()
    
    results_df = pd.DataFrame({
        'Letter': predicted_letter_counts.index,
        'Total Predictions': predicted_letter_counts.values,
        'Accurate Predictions': accurate_predicted_letter_counts.reindex(predicted_letter_counts.index, fill_value=0).values
    })
    
    results_df['Accuracy'] = results_df['Accurate Predictions'] / results_df['Total Predictions']
    results_df_sorted = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
    # Save the results to a CSV file
    results_df_sorted.to_csv(f'sorted_predictions_results_{name}.csv', index=False)
    
    return results_df_sorted

# Process each dataset and store the results in a dictionary
all_results = {}
for name, path in datasets.items():
    all_results[name] = process_dataset(name, path)

# Optionally, print out the results for each dataset
for name, df in all_results.items():
    print(f"Results for {name}:")
    print(df)
    print("\n")

# Save all results to a single Excel file with separate sheets
try:
    with pd.ExcelWriter('all_sorted_predictions_results.xlsx', engine='openpyxl') as writer:
        for name, df in all_results.items():
            df.to_excel(writer, sheet_name=name, index=False)
except ImportError as e:
    print(f"Error: {e}")
    print("Please install openpyxl to save the results to an Excel file. You can install it using 'pip install openpyxl'.")

