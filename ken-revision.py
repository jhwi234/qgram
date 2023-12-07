import logging
import random
import re
import heapq
from pathlib import Path

import kenlm
import numpy as np
import nltk
import subprocess
from retry import retry

logging.basicConfig(level=logging.INFO)
clean_pattern = re.compile(r'\b[a-zA-Z]{3,}\b')

@retry(subprocess.CalledProcessError, tries=3, delay=2)
def model_task(corpus_name, q, corpus_path, model_directory):
    arpa_file = model_directory / f'{corpus_name}_{q}gram.arpa'
    binary_file = model_directory / f'{corpus_name}_{q}gram.klm'

    try:
        subprocess.run(
            ['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ['build_binary', '-s', str(arpa_file), str(binary_file)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return q, str(binary_file)
    except subprocess.CalledProcessError as e:
        logging.error(f"Attempt to generate/load the model for {q}-gram failed: {e}")
        raise

class LanguageModel:
    def __init__(self, q_range=(2, 6), split_config=0.2):
        self.q_range = range(q_range[0], q_range[1] + 1)
        self.model = {}
        self.corpus = set()
        self.test_set = set()
        self.training_set = set()
        self.split_config = split_config

    def clean_text(self, text: str) -> set[str]:
        return set(clean_pattern.findall(text))

    def load_corpus(self):
        # Load the CMU corpus and return a set of words
        nltk.download('cmudict')
        cmu_words = nltk.corpus.cmudict.words()
        cmu_words = [word.lower() for word in cmu_words]
        self.corpus = self.clean_text(' '.join(cmu_words))
        return self.corpus

    def prepare_test_set(self, data, test_size_ratio=0.2):
        total_size = len(data)
        test_size = int(total_size * test_size_ratio)
        random.shuffle(data)
        self.training_set = data[test_size:]
        self.test_set = {self.replace_random_letter(seq) for seq in data[:test_size]}

    def replace_random_letter(self, word):
        letter_index = np.random.randint(0, len(word))
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index+1:]
        return modified_word, missing_letter

    def generate_formatted_corpus(self, path='cmu_formatted_corpus.txt'):
        formatted_text = [" ".join(word) for word in self.corpus]
        formatted_corpus = '\n'.join(formatted_text)
        corpus_path = Path(path)
        with corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_corpus)
        return path
    
    def generate_and_load_models(self, corpus_path):
        model_directory = Path('cmu_models')
        model_directory.mkdir(parents=True, exist_ok=True)

        for q in self.q_range:
            # Assume model_task function exists to create and load models
            _, binary_file = model_task('cmu', q, corpus_path, model_directory)
            if binary_file:
                self.model[q] = kenlm.Model(binary_file)
                logging.info(f"Model for {q}-gram loaded.")

    def predict_missing_letter(self, oov_word):
        missing_letter_index = oov_word.index('_')
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}
        entropy_weights = []
        boundary_start = '<w> ' if missing_letter_index == 0 else ''
        boundary_end = ' </w>' if missing_letter_index == len(oov_word) - 1 else ''
        oov_word_with_boundaries = f"{boundary_start}{oov_word}{boundary_end}"

        for q in self.q_range:
            if q not in self.model:
                print(f"No model found for {q}-grams.")
                continue
            model = self.model[q]
            # Prepare contexts based on the current q value, ensuring not to exceed bounds
            left_size = min(missing_letter_index, q - 1)
            right_size = min(len(oov_word) - missing_letter_index - 1, q - 1)
            left_context = oov_word_with_boundaries[max(0, missing_letter_index - left_size + len(boundary_start)):missing_letter_index + len(boundary_start)]
            right_context = oov_word_with_boundaries[missing_letter_index + len(boundary_start) + 1:missing_letter_index + len(boundary_start) + 1 + right_size]
            # Ensure there are no extra spaces before or after the context
            left_context = left_context.strip()
            right_context = right_context.strip()
            # Joining contexts with spaces as they would appear in the corpus
            left_context_joined = ' '.join(left_context)
            right_context_joined = ' '.join(right_context)
            # Calculate entropy for the current context
            entropy = -sum(model.score(left_context_joined + ' ' + c + ' ' + right_context_joined)
                        for c in 'abcdefghijklmnopqrstuvwxyzæœ')
            entropy_weights.append(entropy)
            for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
                # Create the full sequence with the candidate letter filled in
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                # Get the log score for the full sequence
                log_prob_full = model.score(full_sequence)
                log_probabilities[letter].append(log_prob_full)
        # Normalize entropy weights
        entropy_weights = np.exp(entropy_weights - np.max(entropy_weights))
        entropy_weights /= entropy_weights.sum()
        # Now average the log probabilities across all q values with entropy weights
        averaged_log_probabilities = {}
        for letter, log_probs_list in log_probabilities.items():
            if log_probs_list:
                # Weighted sum of log probabilities using entropy weights
                weighted_log_probs = np.sum([entropy_weights[i] * log_probs
                                            for i, log_probs in enumerate(log_probs_list)], axis=0)
                averaged_log_probabilities[letter] = weighted_log_probs
        # Get the top prediction
        top_prediction = heapq.nlargest(1, averaged_log_probabilities.items(), key=lambda item: item[1])[0]
        top_letter = top_prediction[0]
        top_probability = np.exp(top_prediction[1])
        
        return top_letter, top_probability

    def evaluate_model(self, output_file):
        predictions = []
        correct_predictions = 0
        for modified_word, missing_letter in self.test_set:
            predicted_letter, prediction_confidence = self.predict_missing_letter(modified_word)
            predictions.append((modified_word, missing_letter, predicted_letter, prediction_confidence))
            if predicted_letter == missing_letter:
                correct_predictions += 1

        self.save_predictions_to_file(predictions, output_file)
        accuracy = correct_predictions / len(self.test_set)
        return accuracy

    def save_predictions_to_file(self, predictions, file_name):
        """Save the top prediction result to a file."""
        with open(file_name, 'w') as file:
            for seq, correct_prediction, prediction, prediction_confidence in predictions:
                file.write(f"Test Word: {seq}\nCorrect Prediction: {correct_prediction}\nPredicted Letter: {prediction}\n")
                file.write(f"Confidence: {prediction_confidence:.5f}\n\n")

    def save_set_to_file(self, data_set, file_name):
        """Save a set of data to a file."""
        with open(file_name, 'w') as file:
            for item in data_set:
                file.write(f"{item}\n")

def main():
    print("Main function started")  # Debugging print

    # Initialize the language model
    lm = LanguageModel()
    print("Language model initialized")  # Debugging print

    # Load the CMU corpus
    lm.load_corpus()
    print("Corpus loaded")  # Debugging print

    # Call prepare_test_set
    lm.prepare_test_set(list(lm.corpus))  # Convert to list if needed
    print("Test set prepared")  # Debugging print

    # Save the training set to a file
    lm.save_set_to_file(lm.training_set, "cmu_training_set.txt")

    # Generate a formatted version of the CMU corpus
    corpus_path = lm.generate_formatted_corpus()

    # Generate and load q-gram models
    lm.generate_and_load_models(corpus_path)

    # Prepare file paths for results
    predictions_file = "cmu_predictions.txt"
    test_set_file = "cmu_test_set.txt"
    corpus_file = "cmu_corpus.txt"

    # Save the test set and corpus to files
    lm.save_set_to_file(lm.test_set, test_set_file)
    lm.save_set_to_file(lm.corpus, corpus_file)

    # Evaluate the model and save predictions to a file
    accuracy = lm.evaluate_model(predictions_file)
    logging.info(f"Model accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
