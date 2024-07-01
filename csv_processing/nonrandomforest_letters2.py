import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Function to create binary features for each letter in Top1_Predicted_Letter
def create_top1_letter_features(df):
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for letter in letters:
        df[f'Top1_{letter}'] = df['Top1_Predicted_Letter'].apply(lambda x: 1 if x == letter else 0)
    return df

# Function to calculate accuracy for each letter
def calculate_letter_accuracies(df):
    accuracies = {}
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for letter in letters:
        letter_df = df[df['Top1_Predicted_Letter'] == letter]
        if not letter_df.empty:
            accuracy = letter_df['Top1_Is_Accurate'].mean()
            accuracies[letter] = accuracy
    return pd.DataFrame(list(accuracies.items()), columns=['Letter', 'Accuracy']).sort_values(by='Accuracy', ascending=False)

# Function to run the analysis for a given dataset, focusing on accuracies of individual letters
def run_accuracy_analysis(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data: {dataset_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing file: {dataset_path}")
        return None

    df = create_top1_letter_features(df)
    accuracy_df = calculate_letter_accuracies(df)
    
    return accuracy_df

def aggregate_accuracies(accuracy_dict, top_letters):
    top_accuracies = {'Letter': top_letters}
    for dataset_name, accuracy_df in accuracy_dict.items():
        if accuracy_df is not None:
            top_accuracies[dataset_name] = accuracy_df.set_index('Letter').reindex(top_letters)['Accuracy'].values
    return pd.DataFrame(top_accuracies)

def plot_accuracies(melted_df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=melted_df, x='Letter', y='Accuracy', hue='Dataset')
    plt.title('Accuracy of Top Letters Across Datasets')
    plt.xlabel('Letter')
    plt.ylabel('Accuracy')
    plt.legend(title='Dataset')
    plt.show()

def perform_anova_on_accuracies(top_accuracies, letters):
    accuracy_scores = [top_accuracies.loc[top_accuracies['Letter'] == letter, top_accuracies.columns[1:]].values.flatten() for letter in letters]
    f_statistic, p_value = f_oneway(*accuracy_scores)
    print(f'ANOVA results for top letters {letters}: F-statistic = {f_statistic}, p-value = {p_value}')
    if p_value < 0.05:
        print("The differences in accuracy scores for top letters are statistically significant.")
    else:
        print("The differences in accuracy scores for top letters are not statistically significant.")

# Define dataset paths
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Run analysis for each dataset
accuracy_dict = {}
combined_df = pd.DataFrame()

for name, path in dataset_paths.items():
    print(f"Running accuracy analysis for {name} dataset...")
    accuracy_df = run_accuracy_analysis(path)
    if accuracy_df is not None:
        accuracy_dict[name] = accuracy_df
        print(f"Letter accuracy for {name} dataset:")
        print(accuracy_df)
        print("\n" + "="*80 + "\n")
        
        # Combine datasets for overall analysis
        df = pd.read_csv(path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Define top letters to analyze based on accuracies
top_letters = ['q', 'n', 'g', 't', 'r', 'l', 's']

# Aggregate accuracies across all datasets
top_accuracies_df = aggregate_accuracies(accuracy_dict, top_letters)

# Melt the DataFrame for easier plotting
melted_df = top_accuracies_df.melt(id_vars='Letter', var_name='Dataset', value_name='Accuracy')

# Plot the accuracies
plot_accuracies(melted_df)

# Perform ANOVA to compare accuracy scores for 'q', 'n', and 'g'
perform_anova_on_accuracies(top_accuracies_df, ['q', 'n', 'g'])

# Analyze combined dataset
combined_df = create_top1_letter_features(combined_df)
combined_accuracy_df = calculate_letter_accuracies(combined_df)

print("Combined dataset letter accuracy:")
print(combined_accuracy_df)

# Plot the accuracies for combined dataset
plt.figure(figsize=(12, 8))
sns.barplot(data=combined_accuracy_df, x='Letter', y='Accuracy')
plt.title('Accuracy of Letters in Combined Dataset')
plt.xlabel('Letter')
plt.ylabel('Accuracy')
plt.show()
