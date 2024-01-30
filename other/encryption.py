import random
import nltk
import nltk.corpus
import string
import kenlm
import subprocess
from pathlib import Path
from collections import defaultdict

def simple_substitution_cipher(text, seed=42):
    # Initialize the random generator with a seed for reproducibility
    random.seed(seed)
    letters = 'abcdefghijklmnopqrstuvwxyz'  # Alphabet
    shuffled_letters = list(letters)
    random.shuffle(shuffled_letters)  # Shuffle the letters randomly
    
    # Perform additional shuffling
    for _ in range(5):
        for i in range(len(letters)):
            swap_index = random.randint(0, 25)
            shuffled_letters[i], shuffled_letters[swap_index] = shuffled_letters[swap_index], shuffled_letters[i]

    # Create a mapping from original to cipher letters
    cipher_map = dict(zip(letters, shuffled_letters))
    # Encrypt the text using the cipher map
    encrypted_text = ''.join(cipher_map.get(char, char) for char in text)
    return encrypted_text, cipher_map

# Function to extract and encrypt text from the Brown corpus
def extract_and_encrypt_text(corpus, word_count=400, seed=42):
    # Download the Brown corpus
    nltk.download('brown', quiet=True)
    # Extract words from the corpus
    words = corpus.words()
    # Select a random start point in the corpus
    start = random.randint(0, len(words) - word_count)
    # Extract the specified number of words
    selected_words = words[start:start + word_count]
    # Process and encrypt the text
    tokens = [word.lower() for word in selected_words if word.isalpha()]
    selected_text = ' '.join(tokens)
    encrypted_text, cipher_map = simple_substitution_cipher(selected_text, seed)
    return encrypted_text, cipher_map

# Function to build a KenLM language model
def build_kenlm_model(text, q=6):
    # Create a directory for the KenLM models
    model_directory = Path("./kenlm_models")
    model_directory.mkdir(exist_ok=True)
    # Define file paths for the corpus and model files
    corpus_path = model_directory / "training_corpus.txt"
    arpa_file = model_directory / "model.arpa"
    binary_file = model_directory / "model.klm"

    # Write the text to the corpus file
    with open(corpus_path, "w") as file:
        file.write(text)

    # Run KenLM commands to build the language model
    try:
        subprocess.run(['lmplz', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file), '--discount_fallback'], check=True)
        subprocess.run(['build_binary', '-s', str(arpa_file), str(binary_file)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during model building: {e}")

    # Check the integrity of the model file
    if not binary_file.exists() or binary_file.stat().st_size == 0:
        raise RuntimeError("Binary model file is missing or empty.")

    # Return the KenLM model
    return kenlm.Model(str(binary_file))

# Class for evaluating predictions
class PredictionsEvaluator:
    def __init__(self, model, test_text, cipher_map):
        # Initialize with model, test text, and the cipher mapping
        self.model = model
        self.test_text = test_text
        self.unique_characters = set(test_text)
        self.cipher_map = cipher_map
        # Track occurrences and prediction accuracy
        self.actual_occurrences = defaultdict(int)
        self.correct_predictions = defaultdict(int)
        self.total_predictions = defaultdict(int)

    def evaluate(self):
        # Evaluate each character in the test text
        for i, actual_char in enumerate(self.test_text):
            if actual_char.isalpha():
                # Update occurrences and make predictions
                self.actual_occurrences[actual_char] += 1
                context = self.test_text[:i] + '_' + self.test_text[i+1:]
                predicted_char = self.predict_char(context)
                self.total_predictions[predicted_char] += 1
                if predicted_char == actual_char:
                    self.correct_predictions[actual_char] += 1

        # Calculate precisionand recall for each character
        precision = {char: self.correct_predictions[char] / self.total_predictions[char] if self.total_predictions[char] > 0 else 0 for char in self.unique_characters}
        recall = {char: self.correct_predictions[char] / self.actual_occurrences[char] if self.actual_occurrences[char] > 0 else 0 for char in self.unique_characters}
        return precision, recall

    def predict_char(self, context):
        # Predict the character based on the context using the KenLM model
        best_char = ''
        best_score = float('-inf')
        for char in self.unique_characters:
            test_seq = context.replace('_', char)
            score = self.model.score(test_seq)
            if score > best_score:
                best_char = char
                best_score = score
        return self.cipher_map.get(best_char, best_char)

# Main execution
def main():
    # Extract and encrypt text from the Brown corpus
    encrypted_text, cipher_map = extract_and_encrypt_text(nltk.corpus.brown)

    # Split the encrypted text into training and testing parts
    split_index = len(encrypted_text) // 2
    training_text = encrypted_text[:split_index]
    testing_text = encrypted_text[split_index:]

    # Build a KenLM language model from the training text
    model = build_kenlm_model(training_text)

    # Initialize the evaluator with the model, testing text, and cipher map
    evaluator = PredictionsEvaluator(model, testing_text, cipher_map)
    # Evaluate precision and recall
    precision, recall = evaluator.evaluate()

    # Print the evaluation results
    for char in sorted(precision.keys()):
        print(f"Character: {char}, Precision: {precision[char]:.4f}, Recall: {recall[char]:.4f}")

if __name__ == "__main__":
    main()
