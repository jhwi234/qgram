import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# File paths
clmet_file_path = Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv')
brown_file_path = Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
cmudict_file_path = Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv')

# Assuming you have already defined 'clmet_file_path', 'brown_file_path', and 'cmudict_file_path'
datasets = {
    "CLMET3": pd.read_csv(clmet_file_path),
    "Brown": pd.read_csv(brown_file_path),
    "CMUDict": pd.read_csv(cmudict_file_path)
}

# Prepare data for plotting
for dataset_name, dataset in datasets.items():
    dataset['Word_Length'] = dataset['Original_Word'].fillna('').apply(len)
    dataset['Dataset'] = dataset_name

# Combine all datasets into a single DataFrame for convenience
combined_data = pd.concat(datasets.values())

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
markers = ['o', 's', '^']
palette = sns.color_palette("colorblind", 3)

# Plot each dataset with enhancements
for idx, (dataset_name, df) in enumerate(filtered_grouped_data.groupby('Dataset')):
    # Sort by word length for a proper line plot
    df_sorted = df.sort_values('Word_Length')
    
    # Plot with markers and improved line visibility
    plt.plot(df_sorted['Word_Length'], df_sorted['Accuracy_Mean'], label=dataset_name, color=palette[idx], 
             marker=markers[idx], linewidth=2, alpha=0.75, markersize=8)

    # Enhanced scatter plot to visualize data density
    plt.scatter(df_sorted['Word_Length'], df_sorted['Accuracy_Mean'], color=palette[idx], 
                alpha=0.3, edgecolor='w', s=df_sorted['Sample_Count'])

plt.title('Enhanced Prediction Accuracy vs. Word Length (Up to 15 Characters)')
plt.xlabel('Word Length')
plt.ylabel('Mean Prediction Accuracy')
plt.legend(title='Dataset', loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
sns.despine()
plt.tight_layout()
plt.show()