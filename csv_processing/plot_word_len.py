import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Corrected datasets loading and preparation
datasets_paths = {
    "CLMET3": 'data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv',
    "Lampeter": 'data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv',
    "Edges": 'data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv',
    "Brown": 'data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv'
}

# Load datasets and prepare data for plotting
loaded_datasets = {}  # Dictionary to hold the loaded and processed DataFrames
for dataset_name, file_path in datasets_paths.items():
    dataset = pd.read_csv(file_path)  # Load the dataset
    dataset['Word_Length'] = dataset['Original_Word'].fillna('').apply(len)  # Calculate word length
    dataset['Dataset'] = dataset_name  # Assign dataset name
    loaded_datasets[dataset_name] = dataset  # Store the processed DataFrame

# Combine all datasets into a single DataFrame for convenience
combined_data = pd.concat(loaded_datasets.values())

# Group by dataset and word length to calculate mean accuracy and count
grouped_data = combined_data.groupby(['Dataset', 'Word_Length']).agg(
    Accuracy_Mean=('Top1_Is_Accurate', 'mean'),
    Sample_Count=('Top1_Is_Accurate', 'count')
).reset_index()

# Filter out data with word length greater than 15 for clearer visualization
filtered_grouped_data = grouped_data[grouped_data['Word_Length'] <= 15]

# Create the improved plot with enhancements
plt.figure(figsize=(12, 8))

# Define markers for each dataset for better differentiation and update palette for better visibility
markers = ['o', 's', '^', '>']
palette = sns.color_palette("colorblind", len(datasets_paths))

# Plot each dataset with enhancements
for idx, (dataset_name, df) in enumerate(filtered_grouped_data.groupby('Dataset')):
    # Sort by word length for a proper line plot
    df_sorted = df.sort_values('Word_Length')
    
    # Plot with markers and improved line visibility
    plt.plot(df_sorted['Word_Length'], df_sorted['Accuracy_Mean'], label=dataset_name, color=palette[idx], 
             marker=markers[idx % len(markers)], linewidth=2, alpha=0.75, markersize=8)

    # Enhanced scatter plot to visualize data density
    plt.scatter(df_sorted['Word_Length'], df_sorted['Accuracy_Mean'], color=palette[idx], 
                alpha=0.25, edgecolor='w', s=df_sorted['Sample_Count'])

plt.title('Enhanced Prediction Accuracy vs. Word Length (Up to 15 Characters)')
plt.xlabel('Word Length')
plt.ylabel('Mean Prediction Accuracy')
plt.legend(title='Dataset', loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
sns.despine()
plt.tight_layout()
plt.show()
