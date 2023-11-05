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

np.random.seed(42)
# Compile the regular expression once and use it inside clean method
split_pattern = re.compile(r"[-\s]")

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
        parts = split_pattern.split(word)
        return [
            ''.join(char.lower() for char in part if char.isalpha())
            for part in parts if len(part) >= 4
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
            self.training_corpora[corpus_name] -= test_words
            self.test_set[corpus_name] = {
                word: self.replace_random_letter(word) for word in test_words
            }

    @lru_cache(maxsize=10000)
    def cached_score(self, model, sequence):
        return model.score(sequence, bos=False, eos=False)

    def generate_formatted_corpus(self, corpus_name, path='formatted_corpus.txt', use_test_set=True):
        # Check for cached corpus first
        if corpus_name in self.formatted_corpora_cache:
            return self.formatted_corpora_cache[corpus_name]

        # Helper function to format the corpus
        def format_corpus(words):
            return ' '.join(f'<w> {" ".join(word)} </w>' for word in words)

        # Determine which set of words to format
        words_to_format = self.corpora[corpus_name]
        if use_test_set and corpus_name in self.test_set:
            words_to_format = words_to_format - set(self.test_set[corpus_name].values())

        # Check if the file exists and read from it if so
        corpus_path = Path(path)
        if corpus_path.exists():
            with corpus_path.open('r') as f:
                formatted_corpus = f.read()
        else:
            # File doesn't exist, so format the corpus and write it to a new file
            formatted_corpus = format_corpus(words_to_format)
            with corpus_path.open('w') as f:
                f.write(formatted_corpus)

        # Cache and return the formatted corpus
        self.formatted_corpora_cache[corpus_name] = formatted_corpus
        return path

    def generate_and_load_models(self, corpus_name, corpus_path):
        self.models[corpus_name] = {}
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
        entropy_weights = []

        # Adjust boundary symbols if the missing letter is at the beginning or end of the word
        boundary_start = '<w> ' if missing_letter_index == 0 else ''
        boundary_end = ' </w>' if missing_letter_index == len(oov_word) - 1 else ''
        oov_word_with_boundaries = f"{boundary_start}{oov_word}{boundary_end}"

        for q in self.q_range:
            if q not in self.models[corpus_name]:
                print(f"No model found for {q}-grams in {corpus_name} corpus.")
                continue
            model = self.models[corpus_name][q]

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
            entropy = -sum(self.cached_score(model, left_context_joined + ' ' + c + ' ' + right_context_joined)
                        for c in 'abcdefghijklmnopqrstuvwxyz')
            entropy_weights.append(entropy)

            for letter in 'abcdefghijklmnopqrstuvwxyz':
                # Create sequences for before and after the missing letter
                sequence_before = f"{left_context_joined} {letter}" if left_context_joined else letter
                sequence_after = f"{letter} {right_context_joined}" if right_context_joined else letter

                # Get the log scores from the cached method instead of the model directly
                log_prob_before = self.cached_score(model, sequence_before)
                log_prob_after = self.cached_score(model, sequence_after)

                log_probabilities[letter].append((log_prob_before, log_prob_after))

        # Normalize entropy weights
        entropy_weights = np.exp(entropy_weights - np.max(entropy_weights))
        entropy_weights /= entropy_weights.sum()

        # Now average the log probabilities across all q values with entropy weights
        averaged_log_probabilities = {}
        for letter, log_probs_list in log_probabilities.items():
            if log_probs_list:
                # Weighted sum of log probabilities using entropy weights
                weighted_log_probs = np.sum([entropy_weights[i] * (log_probs[0] + log_probs[1]) / 2
                                            for i, log_probs in enumerate(log_probs_list)], axis=0)
                averaged_log_probabilities[letter] = weighted_log_probs

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
            correct = {1: 0, 2: 0, 3: 0}
            directory = Path(f"./results/{corpus_name}")
            directory.mkdir(parents=True, exist_ok=True)

            result_lines = []
            for original_word, test_word in test_words.items():
                predictions = self.predict_missing_letter(corpus_name, test_word)
                correct_letter = original_word[test_word.index('_')]
                
                # Check if the correct prediction is in top 1, top 2, and top 3
                top_predictions = [p[0] for p in predictions]
                if correct_letter == top_predictions[0]:
                    correct[1] += 1
                if correct_letter in top_predictions[:2]:
                    correct[2] += 1
                if correct_letter in top_predictions[:3]:
                    correct[3] += 1

                result_lines.append(
                    f"Iteration: {iteration_number}\n"
                    f"Test Word: {test_word}\n"
                    f"Correct Word: {original_word}\n"
                    "Top 3 Predictions:\n" +
                    "".join(f"Rank {rank}: '{letter}' with probability {prob:.5f}\n"
                            for rank, (letter, prob) in enumerate(predictions, 1) if rank < 4)
                )

            with (directory / f"results_{today}.txt").open('a') as f:
                f.write("\n\n".join(result_lines) + "\n\n")

            # Calculate accuracies
            total_words = len(test_words)
            results[corpus_name] = {
                'top1': correct[1] / total_words,
                'top2': correct[2] / total_words,
                'top3': correct[3] / total_words,
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
    print(f"Word: {word}")
    
    # Calculate total probability of top 5 to normalize and show as percentages
    total_prob = sum(prob for _, prob in predictions)
    if total_prob == 0:
        print("No predictions to display.")
        return

    for rank, (letter, prob) in enumerate(predictions, 1):
        percentage = (prob / total_prob) * 100  # Convert to percentage of the total
        print(f"Rank {rank}: '{letter}' with {percentage:.2f}% of the top 3 confidence")

if __name__ == "__main__":
    iterations = 10
    corpora = ['brown', 'cmu']
    total_accuracy = {corpus_name: {'top1': 0, 'top2': 0, 'top3': 0} for corpus_name in corpora}
    today = datetime.now().strftime('%Y%m%d')

    # Initialize LanguageModel outside the loop to keep the corpora loaded
    lm = LanguageModel(q_range=(2, 6))
    lm.load_corpora(use_test_set=False)

    formatted_corpus_paths = {
        corpus_name: lm.generate_formatted_corpus(corpus_name, path=f'{corpus_name}_formatted_corpus.txt')
        for corpus_name in corpora
    }

    for iteration in range(iterations):
        lm.prepare_test_set()
        
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