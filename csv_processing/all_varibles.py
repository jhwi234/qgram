import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    df['Word_Length'] = df['Original_Word'].apply(lambda x: len(str(x)))
    df['Missing_Letter_Index'] = df.apply(lambda row: str(row['Original_Word']).find("_"), axis=1)
    df['Relative_Position'] = df.apply(lambda row: row['Missing_Letter_Index'] / (row['Word_Length'] - 1) if row['Word_Length'] > 1 else 0, axis=1)
    df['Top1_Predicted_Letter_is_Vowel'] = df['Top1_Predicted_Letter'].apply(lambda x: x.lower() in ['a', 'e', 'i', 'o', 'u'])
    df['Correct_Letter_is_Vowel'] = df['Correct_Letter'].apply(lambda x: x.lower() in ['a', 'e', 'i', 'o', 'u'])
    
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
    features, target = load_and_preprocess(path)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    feature_importances = clf.feature_importances_
    
    results[title] = {
        "accuracy": accuracy,
        "feature_importances": feature_importances
    }

# Define the feature names to make the output more readable
feature_names = ['Word Length', 'Relative Position', 'Top1 Predicted Letter is Vowel', 'Correct Letter is Vowel']

for title, info in results.items():
    formatted_feature_importances = [f"{importance:.4f}" for importance in info['feature_importances']]
    print(f"{title}:")
    print(f"Accuracy = {info['accuracy']:.2f}")
    print("Feature Importances")
    for name, formatted_importance in zip(feature_names, formatted_feature_importances):
        print(f"    {name}: {formatted_importance}")
    print()  # Blank line for better separation