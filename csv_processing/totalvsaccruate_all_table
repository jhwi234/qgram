import pandas as pd

# Define dataset paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Combine all datasets into one DataFrame
combined_data = pd.DataFrame()

for name, path in datasets.items():
    data = pd.read_csv(path)
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Calculate total and accurate predictions for each letter
predicted_letter_counts = combined_data['Top1_Predicted_Letter'].value_counts().sort_index()
accurate_predicted_letter_counts = combined_data[combined_data['Top1_Is_Accurate']]['Top1_Predicted_Letter'].value_counts().sort_index()

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Letter': predicted_letter_counts.index,
    'Total Predictions': predicted_letter_counts.values,
    'Accurate Predictions': accurate_predicted_letter_counts.reindex(predicted_letter_counts.index, fill_value=0).values
})

# Calculate accuracy
results_df['Accuracy'] = results_df['Accurate Predictions'] / results_df['Total Predictions']

# Sort the results DataFrame by accuracy in descending order
results_df_sorted = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

# Display the sorted table
print(results_df_sorted)

# Optionally, save the results to a CSV file
results_df_sorted.to_csv('sorted_combined_predictions_results.csv', index=False)
