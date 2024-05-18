# Optimized Python code with enhanced comments and some structural improvements.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

def is_vowel(letter):
    """Check if a letter is a vowel. Including 'y' as a sometimes vowel."""
    return letter.lower() in 'aeiouy'

def load_and_preprocess(path):
    """Load the dataset and preprocess it for model training.
    
    Args:
        path (str or Path): Path to the dataset.

    Returns:
        tuple: A tuple containing the features DataFrame and target series.
    """
    df = pd.read_csv(path)
    df['Word_Length'] = df['Original_Word'].str.len() # Create a new column for word length
    df['Missing_Letter_Index'] = df['Original_Word'].str.find("_") # Find the index of the missing letter
    df['Relative_Position'] = df['Missing_Letter_Index'] / df['Word_Length'].clip(lower=1)  # Avoid division by zero
    df['Top1_Predicted_Letter_is_Vowel'] = df['Top1_Predicted_Letter'].apply(is_vowel) # Check if the predicted letter is a vowel
    df['Correct_Letter_is_Vowel'] = df['Correct_Letter(s)'].apply(is_vowel) # Check if the correct letter is a vowel

    features = df[['Word_Length', 'Relative_Position', 'Top1_Predicted_Letter_is_Vowel', 'Correct_Letter_is_Vowel']]
    target = df['Top1_Is_Accurate']

    return features, target

def train_and_evaluate_model(features, target):
    """Train and evaluate the Random Forest model.

    Args:
        features (DataFrame): The features for model training.
        target (Series): The target variable for the model.

    Returns:
        tuple: A tuple containing the model's accuracy and feature importances.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    feature_importances = clf.feature_importances_

    return accuracy, feature_importances

def main():
    dataset_paths = {
        "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
    }

    results = {}
    feature_names = ['Word Length', 'Relative Position', 'Predicted Letter is Vowel', 'Correct Letter is Vowel']

    for title, path in dataset_paths.items():
        print(f"Processing dataset: {title}")
        features, target = load_and_preprocess(path)
        accuracy, feature_importances = train_and_evaluate_model(features, target)
        results[title] = {"accuracy": accuracy, "feature_importances": feature_importances}

    for title, info in results.items():
        print(f"\n{title}:")
        print(f"Accuracy = {info['accuracy']:.4f}")
        print("Feature Importances:")
        for name, importance in zip(feature_names, info['feature_importances']):
            print(f"    {name}: {importance:.4f}")

if __name__ == "__main__":
    main()

