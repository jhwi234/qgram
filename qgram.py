import nltk
import subprocess
import kenlm
import heapq
from functools import lru_cache
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import logging

logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)

class LanguageModel:

    def __init__(self, q_range=(2, 6)):
        self.q_range = range(q_range[0], q_range[1] + 1)
        self.models = {}
        self.corpora = {
            'brown': set(),
            'cmu': set(),
        }
        self.formatted_corpora_cache = {}
        self.test_set = {}
        self.training_corpora = {}
        self.loaded_corpora = False 

    nltk_resources_downloaded = False

    @classmethod
    def download_nltk(cls):
        if not cls.nltk_resources_downloaded:
            nltk_resources = [
                'corpora/cmudict', 'tokenizers/punkt', 'corpora/brown'
            ]
            for resource in nltk_resources:
                try:
                    nltk.data.find(resource)
                except LookupError:
                    nltk.download(resource.split('/')[-1])
            # Set the flag to True after downloading
            cls.nltk_resources_downloaded = True

    def clean_word(self, word):
        return [
            ''.join(filter(str.isalpha, part)).lower()
            for part in re.split(r"[-\s]", word)
        ]

    def load_corpora(self, use_test_set=True):
        if not self.loaded_corpora:
            self.download_nltk()
            self.corpora['cmu'].update(self.load_cmu())
            self.corpora['brown'].update(self.load_brown())
            self.loaded_corpora = True

        if use_test_set:
            self.prepare_test_set() 

    def load_cmu(self):
        cmu_dict = nltk.corpus.cmudict.dict()
        return {
            cleaned_word
            for word in cmu_dict.keys() if word.isalpha()
            for cleaned_word in self.clean_word(word)
        }

    def load_brown(self):
        brown_words = nltk.corpus.brown.words()
        return {
            cleaned_word
            for word in brown_words
            for cleaned_word in self.clean_word(word)
            if cleaned_word
        }

    @staticmethod
    def replace_random_letter(word):
        """Replace a random letter in a word with an underscore '_'"""
        letter_index = np.random.randint(0, len(word))
        return word[:letter_index] + '_' + word[letter_index+1:]

    def prepare_test_set(self, n=100):
        """
        Prepare test set by selecting n words from the corpora
        and replacing a random letter with an underscore '_'.
        Does not remove words from the original corpora.
        """
        self.test_set = {}
        self.training_corpora = {corpus: set(words) for corpus, words in self.corpora.items()}
        for corpus_name, words in self.corpora.items():
            test_words = set(np.random.choice(list(words), n, replace=False))
            self.training_corpora[corpus_name] -= test_words  # Remove test words from training set copy
            self.test_set[corpus_name] = {
                word: self.replace_random_letter(word) for word in test_words
            }

    @lru_cache(maxsize=10000)
    def cached_score(self, model, sequence):
        return model.score(sequence, bos=False, eos=False)

    def generate_formatted_corpus(self, corpus_name, path='formatted_corpus.txt', use_test_set=True):
        # First, check if the corpus is already cached in memory
        if corpus_name in self.formatted_corpora_cache:
            return self.formatted_corpora_cache[corpus_name]

        # If not cached, proceed to check if the file exists
        if not Path(path).exists():
            # If the file does not exist, create and cache it
            with open(path, 'w') as f:
                formatted_corpus = ' '.join(
                    f'<w> {" ".join(word)} </w>' for word in self.corpora[corpus_name])  # Space between each letter
                f.write(formatted_corpus)
            self.formatted_corpora_cache[corpus_name] = formatted_corpus
        else:
            # If the file exists, read and cache it
            with open(path, 'r') as f:
                formatted_corpus = f.read()

        if use_test_set and corpus_name in self.test_set:
            training_words = self.corpora[corpus_name] - set(self.test_set[corpus_name].values())
            formatted_corpus = ' '.join(
                f'<w> {" ".join(word)} </w>' for word in training_words)  # Space between each letter
        else:
            formatted_corpus = ' '.join(
                f'<w> {" ".join(word)} </w>' for word in self.corpora[corpus_name])  # Space between each letter

        # Cache the corpus content in memory whether it was read from file or written to file
        self.formatted_corpora_cache[corpus_name] = formatted_corpus
        return path

    def generate_and_load_models(self, corpus_name, corpus_path):
        self.models[corpus_name] = {}  # Initialize a dictionary for this corpus
        for q in self.q_range:
            arpa_file = f'{corpus_name}_{q}gram.arpa'
            binary_file = f'{corpus_name}_{q}gram.klm'
            # It's better to use the full path for subprocesses to avoid working directory issues
            full_arpa_path = f'./{arpa_file}'
            full_binary_path = f'./{binary_file}'
            try:
                subprocess.run(['lmplz', '--discount_fallback', '-o', str(q), '--text', corpus_path, '--arpa', full_arpa_path],
                            check=True, capture_output=True)
                # Add the -s flag to build_binary to skip </s> checks
                subprocess.run(['build_binary', '-s', full_arpa_path, full_binary_path],
                                check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e.output.decode()}")
                print(f"Stderr: {e.stderr.decode()}")
                continue

            self.models[corpus_name][q] = kenlm.Model(full_binary_path)
            print(f"Model for {q}-gram loaded for {corpus_name} corpus.")

    def predict_missing_letter(self, corpus_name, oov_word):
        missing_letter_index = oov_word.index('_')
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyz'}

        # Adjust boundary symbols if the missing letter is at the beginning or end of the word
        boundary_start = '<w> ' if missing_letter_index == 0 else ''
        boundary_end = ' </w>' if missing_letter_index == len(oov_word) - 1 else ''
        oov_word_with_boundaries = f"{boundary_start}{oov_word}{boundary_end}"

        for q in self.q_range:
            if q not in self.models[corpus_name]:
                print(f"No model found for {q}-grams in {corpus_name} corpus.")
                continue  # Skip if no model found for the current q value
            model = self.models[corpus_name][q]  # Use the model for the specific q-gram

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

            # Calculate weights based on the amount of context available
            left_weight = len(left_context) / (len(left_context) + len(right_context)) if len(left_context) + len(right_context) > 0 else 0.5
            right_weight = len(right_context) / (len(left_context) + len(right_context)) if len(left_context) + len(right_context) > 0 else 0.5

            for letter in 'abcdefghijklmnopqrstuvwxyz':
                # Create sequences for before and after the missing letter
                sequence_before = f"{left_context_joined} {letter}" if left_context_joined else letter
                sequence_after = f"{letter} {right_context_joined}" if right_context_joined else letter

                # Get the log scores from the cached method instead of the model directly
                log_prob_before = self.cached_score(model, sequence_before)
                log_prob_after = self.cached_score(model, sequence_after)

                # Use the weights to take a weighted average of the log probabilities
                weighted_average_log_prob = (log_prob_before * left_weight + log_prob_after * right_weight)
                log_probabilities[letter].append(weighted_average_log_prob)

        # Now average the log probabilities across all q values
        averaged_log_probabilities = {letter: sum(log_probs)/len(log_probs) for letter, log_probs in log_probabilities.items() if log_probs}

        # Apply heapq.nlargest on log probabilities directly
        top_log_predictions = heapq.nlargest(5, averaged_log_probabilities.items(), key=lambda item: item[1])

        # Convert only the top log probabilities to probabilities
        top_predictions = [(letter, np.exp(log_prob)) for letter, log_prob in top_log_predictions]

        # Return predictions with probabilities
        return top_predictions

    def evaluate_predictions(self, iteration_number):
        today = datetime.now().strftime('%Y%m%d')
        results = {}

        for corpus_name, test_words in self.test_set.items():
            correct = {1: 0, 3: 0, 5: 0}
            directory = Path(f"./results/{corpus_name}")
            directory.mkdir(parents=True, exist_ok=True)

            # Write predictions to a file
            result_lines = []
            for original_word, test_word in test_words.items():
                predictions = self.predict_missing_letter(corpus_name, test_word)
                correct_letter = original_word[test_word.index('_')]
                
                # Check if the correct prediction is in top 1, top 3, and top 5
                top_predictions = [p[0] for p in predictions]
                for k in correct.keys():
                    if correct_letter in top_predictions[:k]:
                        correct[k] += 1

                result_lines.append(
                    f"Iteration: {iteration_number}\n"
                    f"Test Word: {test_word}\n"
                    f"Correct Word: {original_word}\n"
                    "Top 5 Predictions:\n" +
                    "".join(f"Rank {rank}: '{letter}' with probability {prob:.5f}\n"
                            for rank, (letter, prob) in enumerate(predictions, 1))
                )

            # Open the file in append mode ('a') instead of write mode ('w')
            with (directory / f"results_{today}.txt").open('a') as f:
                f.write("\n\n".join(result_lines) + "\n\n")

            # Calculate accuracies
            total_words = len(test_words)
            results[corpus_name] = {
                'top1': correct[1] / total_words,
                'top3': correct[3] / total_words,
                'top5': correct[5] / total_words,
            }

        return results

def save_results_to_file(results, iteration, folder="results"):
    # Ensure the folder exists
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Create a filename with the iteration number and today's date
    today = datetime.now().strftime("%Y%m%d")  # Format for: YYYYMMDD
    filename = f"results_iteration_{iteration}_{today}.txt"
    filepath = folder_path / filename
    
    with filepath.open('w') as file:
        for corpus_name, accuracies in results.items():
            file.write(f"{corpus_name.capitalize()} Corpus:\n")
            for acc_type, acc_value in accuracies.items():
                file.write(f"  {acc_type.upper()} Accuracy: {acc_value:.2%}\n")
            file.write("\n")
    
    print(f"Results saved to {filepath}")

def print_predictions(word: str, predictions: list[tuple[str, float]]) -> None:
    """
    Print predictions for a given word, showing the rank and probability as a percentage of the top 5.
    
    :param word: The word to print predictions for.
    :param predictions: A list of tuples containing predicted letters and their probabilities.
    """
    print(f"Word: {word}")
    
    # Calculate total probability of top 5 to normalize and show as percentages
    total_prob = sum(prob for _, prob in predictions)
    if total_prob == 0:
        print("No predictions to display.")
        return

    for rank, (letter, prob) in enumerate(predictions, 1):
        percentage = (prob / total_prob) * 100  # Convert to percentage of the total
        print(f"Rank {rank}: '{letter}' with {percentage:.2f}% of the top 5 confidence")

if __name__ == "__main__":
    iterations = 2
    corpora = ['brown', 'cmu']
    total_accuracy = {corpus_name: {'top1': 0, 'top3': 0, 'top5': 0} for corpus_name in corpora}
    today = datetime.now().strftime('%Y%m%d')

    # Initialize LanguageModel outside the loop to keep the corpora loaded
    lm = LanguageModel(q_range=(2, 6))
    lm.load_corpora(use_test_set=False)  # Load corpora without preparing the test set

    formatted_corpus_paths = {
        corpus_name: lm.generate_formatted_corpus(corpus_name, path=f'{corpus_name}_formatted_corpus.txt')
        for corpus_name in corpora
    }

    for iteration in range(iterations):
        lm.prepare_test_set()  # Prepare the test set before each iteration
        
        # Load models for each pre-generated formatted corpus
        for corpus_name, corpus_path in formatted_corpus_paths.items():
            lm.generate_and_load_models(corpus_name, corpus_path)
        
        # Evaluate predictions and accumulate accuracies
        accuracies = lm.evaluate_predictions(iteration + 1)
        for corpus_name, accuracy in accuracies.items():
            for acc_type, acc_value in accuracy.items():
                total_accuracy[corpus_name][acc_type] += acc_value
        
        # Print the accuracies for this iteration
        formatted_accuracies = "\n".join(
            f"{corpus_name.capitalize()} Corpus: " +
            ", ".join(f"{acc_type.upper()} Accuracy: {acc_value:.2%}" for acc_type, acc_value in corpus_accuracies.items())
            for corpus_name, corpus_accuracies in accuracies.items()
        )
        print(f"Iteration {iteration + 1} Accuracies:\n{formatted_accuracies}\n")
        
        # Save the results to a file
        save_results_to_file(accuracies, iteration + 1)

    # Calculate averaged accuracy
    averaged_accuracy = {corpus_name: {acc_type: acc_value / iterations
                                       for acc_type, acc_value in acc_types.items()}
                         for corpus_name, acc_types in total_accuracy.items()}
    
    # Print final averaged accuracies
    print("Averaged accuracy over iterations:")
    for corpus_name, acc_types in averaged_accuracy.items():
        accuracies_formatted = "\n".join(f"  {acc_type.upper()} Accuracy: {acc_value:.2%}"
                                         for acc_type, acc_value in acc_types.items())
        print(f"{corpus_name.capitalize()} Corpus:\n{accuracies_formatted}\n")
