import pandas as pd
import matplotlib.pyplot as plt

# Define dataset paths
datasets = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(path) for path in datasets.values()], ignore_index=True)

# Plot the distribution of predicted letters and their accuracies
plt.figure(figsize=(12, 6))

# Count the occurrences of each predicted letter
predicted_letter_counts = combined_data['Top1_Predicted_Letter'].value_counts().sort_index()

# Count the occurrences of accurate predictions for each letter
accurate_predicted_letter_counts = combined_data[combined_data['Top1_Is_Accurate']]['Top1_Predicted_Letter'].value_counts().sort_index()

# Plot total predictions
plt.bar(predicted_letter_counts.index, predicted_letter_counts.values, label='Total Predictions', alpha=0.7)

# Plot accurate predictions
plt.bar(accurate_predicted_letter_counts.index, accurate_predicted_letter_counts.values, label='Accurate Predictions', alpha=0.7)

# Add title and labels
plt.title('Distribution of Predicted Letters and Their Accuracies (Combined Datasets)')
plt.xlabel('Predicted Letter')
plt.ylabel('Number of Predictions')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
