import random
import nltk
import kenlm
import subprocess
from pathlib import Path
from collections import defaultdict

# Function to encrypt text using a simple substitution cipher
def simple_substitution_cipher(text, seed=42):
    random.seed(seed)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    shuffled_letters = list(letters)
    random.shuffle(shuffled_letters)
    
    # Swapping logic
    for _ in range(5):
        for i in range(len(letters)):
            swap_index = random.randint(0, 25)
            shuffled_letters[i], shuffled_letters[swap_index] = shuffled_letters[swap_index], shuffled_letters[i]

    cipher_map = dict(zip(letters, shuffled_letters))
    encrypted_text = ''.join(cipher_map.get(char, char) for char in text)
    return encrypted_text, cipher_map

# Function to extract and encrypt text from the Brown corpus
def extract_and_encrypt_text(corpus, word_count=400, seed=42):
    nltk.download('brown', quiet=True)
    words = corpus.words()
    start = random.randint(0, len(words) - word_count)
    selected_text = ' '.join(words[start:start + word_count])
    encrypted_text, _ = simple_substitution_cipher(selected_text, seed)
    return encrypted_text

# Function to build a KenLM language model
def build_kenlm_model(text, q=6):
    model_directory = Path("./kenlm_models")
    model_directory.mkdir(exist_ok=True)
    corpus_path = model_directory / "training_corpus.txt"
    arpa_file = model_directory / "model.arpa"
    binary_file = model_directory / "model.klm"

    with open(corpus_path, "w") as file:
        file.write(text)

    subprocess.run(['lmplz', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)], check=True)
    subprocess.run(['build_binary', str(arpa_file), str(binary_file)], check=True)

    return kenlm.Model(str(binary_file))

# Class for evaluating predictions
class PredictionsEvaluator:
    def __init__(self, model, test_text, cipher_map):
        self.model = model
        self.test_text = test_text
        self.unique_characters = set(test_text)
        self.cipher_map = cipher_map
        self.actual_occurrences = defaultdict(int)
        self.correct_predictions = defaultdict(int)
        self.total_predictions = defaultdict(int)

    def evaluate(self):
        for i, actual_char in enumerate(self.test_text):
            if actual_char.isalpha():
                self.actual_occurrences[actual_char] += 1
                context = self.test_text[:i] + '_' + self.test_text[i+1:]
                predicted_char = self.predict_char(context)
                self.total_predictions[predicted_char] += 1
                if predicted_char == actual_char:
                    self.correct_predictions[actual_char] += 1

        precision = {char: self.correct_predictions[char] / self.total_predictions[char] if self.total_predictions[char] > 0 else 0 for char in self.unique_characters}
        recall = {char: self.correct_predictions[char] / self.actual_occurrences[char] if self.actual_occurrences[char] > 0 else 0 for char in self.unique_characters}
        return precision, recall

    def predict_char(self, context):
        best_char = ''
        best_score = float('-inf')
        for char in self.unique_characters:
            test_seq = context.replace('_', char)
            # Ensure the sequence length aligns with the KenLM model's n-gram size
            # [Consider trimming or padding the sequence as needed]
            score = self.model.score(test_seq)
            if score > best_score:
                best_char = char
                best_score = score
        # Map the predicted character back to the original character
        return self.cipher_map.get(best_char, best_char)

# Main execution
def main():
    # Extract and encrypt text
    encrypted_text = extract_and_encrypt_text(nltk.corpus.brown)

    # Split text into training and testing parts
    split_index = len(encrypted_text) // 2
    training_text = encrypted_text[:split_index]
    testing_text = encrypted_text[split_index:]

    # Build KenLM model
    model = build_kenlm_model(training_text)

    # Evaluate predictions
    evaluator = PredictionsEvaluator(model, testing_text)
    precision, recall = evaluator.evaluate()

    # Print precision and recall
    for char in sorted(precision.keys()):
        print(f"Character: {char}, Precision: {precision[char]:.4f}, Recall: {recall[char]:.4f}")

if __name__ == "__main__":
    main()
