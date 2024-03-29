import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected datasets loading and preparation
datasets = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange6-6_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange6-6_prediction.csv')
    }

# Update: Load datasets and prepare them
for name, path in datasets.items():
    data = pd.read_csv(path)
    data['Word_Length'] = data['Original_Word'].fillna('').apply(len)
    data['Dataset'] = name
    datasets[name] = data  # Replace path with actual data

# Combine datasets
combined_data = pd.concat(datasets.values())

# Filter for word lengths up to 16
filtered_data = combined_data[combined_data['Word_Length'] <= 15]

# Group and aggregate across all datasets, after filtering
overall_grouped_data = filtered_data.groupby('Word_Length').agg(
    Overall_Accuracy_Mean=('Top1_Is_Accurate', 'mean'),
    Sample_Count=('Top1_Is_Accurate', 'count')
).reset_index()

# Normalize the Sample_Count for color mapping
norm = plt.Normalize(overall_grouped_data['Sample_Count'].min(), overall_grouped_data['Sample_Count'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

# Plotting with explicit fig, ax
fig, ax = plt.subplots(figsize=(12, 8))
for i in range(len(overall_grouped_data) - 1):
    ax.plot(overall_grouped_data['Word_Length'][i:i+2], overall_grouped_data['Overall_Accuracy_Mean'][i:i+2], 
             color=sm.to_rgba(overall_grouped_data['Sample_Count'].iloc[i]), linewidth=2)

# Binding the ScalarMappable to the axes for the colorbar
plt.colorbar(sm, label='Sample Count', ax=ax)

ax.set_title('Overall Prediction Accuracy vs. Word Length with Gradient (Up to 16 Letters)')
ax.set_xlabel('Word Length')
ax.set_ylabel('Mean Prediction Accuracy')
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
sns.despine()
plt.tight_layout()
plt.show()
