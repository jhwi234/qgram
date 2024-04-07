import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

def is_vowel(letter):
    """Check if a letter is a vowel."""
    return letter.lower() in ['a', 'e', 'i', 'o', 'u']

def load_and_preprocess(path):
    """Load the dataset and preprocess it."""
    df = pd.read_csv(path)
    df['Word_Length'] = df['Original_Word'].str.len()
    df['Missing_Letter_Index'] = df['Original_Word'].str.find("_")
    df['Relative_Position'] = df['Missing_Letter_Index'] / (df['Word_Length'] - 1)
    df['Relative_Position'] = df['Relative_Position'].fillna(0)  # Handle division by zero
    df['Top1_Predicted_Letter_is_Vowel'] = df['Top1_Predicted_Letter'].apply(is_vowel)
    df['Correct_Letter_is_Vowel'] = df['Correct_Letter'].apply(is_vowel)
    
    features = df[['Word_Length', 'Relative_Position', 'Top1_Predicted_Letter_is_Vowel', 'Correct_Letter_is_Vowel']]
    target = df['Top1_Is_Accurate']
    
    return features, target

dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange8-8_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange8-8_prediction.csv')
}

results = {}

for title, path in dataset_paths.items():
    print(f"Processing dataset: {title}")
    features, target = load_and_preprocess(path)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    feature_importances = clf.feature_importances_
    
    results[title] = {"accuracy": accuracy, "feature_importances": feature_importances}

# Presenting the results
feature_names = ['Word Length', 'Relative Position', 'Predicted Letter is Vowel', 'Correct Letter is Vowel']

for title, info in results.items():
    print(f"\n{title}:")
    print(f"Accuracy = {info['accuracy']:.2f}")
    print("Feature Importances:")
    for name, importance in zip(feature_names, info['feature_importances']):
        print(f"    {name}: {importance:.4f}")