import io
import hashlib
import heapq
import logging
import random
import re
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import kenlm
import nltk
import numpy as np
from nltk.corpus import brown, cmudict

logging.basicConfig(level=logging.INFO)
split_pattern = re.compile(r"[-\s]")
class LanguageModel:

    def __init__(self, q_range=(2, 6)) -> None:
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

    def clean_word(self, word) -> list[str]:
        if isinstance(word, tuple):
            word = word[0]
        parts = split_pattern.split(word)
        return [
            ''.join(char.lower() for char in part if char.isalpha())
            for part in parts if len(part) >= 4
        ]

    def load_corpora(self, use_test_set: bool = True) -> None:
        if not self.loaded_corpora:
            self.download_nltk_resources()
            self.corpora['cmu'] = self.load_nltk_corpus('cmudict')
            self.corpora['brown'] = self.load_nltk_corpus('brown')
            self.loaded_corpora = True
            if use_test_set:
                self.prepare_test_set()

    def load_nltk_corpus(self, corpus_name: str) -> set[str]:
        nltk.download(corpus_name)
        if corpus_name == 'cmudict':
            entries = getattr(nltk.corpus, corpus_name).entries()
            words = [word for word, _ in entries]
        else:
            words = getattr(nltk.corpus, corpus_name).words()

        return {cleaned_word
                for word in words if isinstance(word, str) and word.isalpha()
                for cleaned_word in self.clean_word(word) if cleaned_word}

    def download_nltk_resources(self) -> None:
        resources = ['cmudict', 'punkt', 'brown']
        for resource in resources:
            resource_id = f'corpora/{resource}'
            try:
                nltk.data.find(resource_id)
            except LookupError:
                logging.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)
        self.__class__.nltk_resources_downloaded = True

    def replace_random_letter(self, word):
        """
        Instance method to replace a random letter in the word with an underscore '_'.
        This method now has access to the instance scope (self), 
        which allows it to use or modify instance variables if needed.
        """
        letter_index = np.random.randint(0, len(word))
        return word[:letter_index] + '_' + word[letter_index+1:]

    def prepare_test_set(self, n=100) -> None:
        """
        Prepare test set by selecting n words from the corpora
        and replacing a random letter with an underscore '_'.
        Does not remove words from the original corpora.
        """
        self.test_set = {}
        self.training_corpora = {corpus: set(words) for corpus, words in self.corpora.items()}
        for corpus_name, words in self.corpora.items():
            test_words = set(random.sample(list(words), n))
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
            words_to_format = words_to_format - set(self.test_set[corpus_name])

        formatted_corpus = format_corpus(words_to_format)
        corpus_hash = hashlib.sha1(formatted_corpus.encode('utf-8')).hexdigest()
        corpus_path = Path(path)

        # Attempt to read the existing file and compare hashes
        try:
            if corpus_path.is_file():
                existing_corpus = corpus_path.read_text()
                existing_hash = hashlib.sha1(existing_corpus.encode('utf-8')).hexdigest()
                if existing_hash == corpus_hash:
                    self.formatted_corpora_cache[corpus_name] = existing_corpus
                    return path

            # Write the updated corpus if the file doesn't exist or the hash doesn't match
            corpus_path.write_text(formatted_corpus)
            self.formatted_corpora_cache[corpus_name] = formatted_corpus
            return path

        except IOError as e:
            logging.error(f"An I/O error occurred: {e}")
            # Handle the error as appropriate
            raise e

    def generate_and_load_models(self, corpus_name, corpus_path):
        self.models[corpus_name] = {}
        model_directory = Path(f'{corpus_name}_models')
        model_directory.mkdir(parents=True, exist_ok=True)

        for q in self.q_range:
            arpa_file = model_directory / f'{corpus_name}_{q}gram.arpa'
            binary_file = model_directory / f'{corpus_name}_{q}gram.klm'

            try:
                subprocess.run(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                            check=True, capture_output=True)
                subprocess.run(['build_binary', '-s', str(arpa_file), str(binary_file)],
                            check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"An error occurred while generating/loading the model for {q}-gram: {e.output.decode()}")
                logging.error(f"Stderr: {e.stderr.decode()}")
                continue

            self.models[corpus_name][q] = kenlm.Model(str(binary_file))
            logging.info(f"Model for {q}-gram loaded for {corpus_name} corpus.")

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
        top_log_predictions = heapq.nlargest(3, averaged_log_probabilities.items(), key=lambda item: item[1])

        # Convert only the top log probabilities to probabilities
        top_predictions = [(letter, np.exp(log_prob)) for letter, log_prob in top_log_predictions]

        # Return predictions with probabilities
        return top_predictions

    def calculate_metrics(self, true_positives, false_positives, false_negatives):
        """
        Calculate precision and recall from true positives, false positives, and false negatives.
        """
        precision = 0.0
        recall = 0.0
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        
        return precision, recall

    def evaluate_predictions(self, iteration_number):
        today = datetime.now().strftime('%Y%m%d')
        results = {}

        for corpus_name, test_words in self.test_set.items():
            correct = {1: 0, 2: 0, 3: 0}
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            result_lines = []

            # Create directory before the loop
            directory = Path(f"./results/{corpus_name}")
            directory.mkdir(parents=True, exist_ok=True)

            for original_word, test_word in test_words.items():
                predictions = self.predict_missing_letter(corpus_name, test_word)
                correct_letter = original_word[test_word.index('_')]

                top_predictions = [p[0] for p in predictions][:3]
                found_at_rank = None

                for rank, pred in enumerate(top_predictions, start=1):
                    if pred == correct_letter:
                        found_at_rank = rank
                        break

                if found_at_rank:
                    for i in range(found_at_rank, 4):
                        correct[i] += 1
                    if found_at_rank == 1:
                        true_positives += 1
                else:
                    false_negatives += 1

                # False positives are only counted when the top prediction is incorrect
                if found_at_rank != 1:
                    false_positives += 1

                # Append results for this word to result_lines
                result_lines.append(
                    f"Iteration: {iteration_number}\n"
                    f"Test Word: {test_word}\n"
                    f"Correct Word: {original_word}\n"
                    "Top 3 Predictions:\n" +
                    "".join(f"Rank {rank}: '{letter}' with probability {prob:.5f}\n"
                            for rank, (letter, prob) in enumerate(predictions, 1) if rank <= 3)
                )

            # Calculate precision and recall
            precision, recall = self.calculate_metrics(true_positives, false_positives, false_negatives)

            # Calculate accuracies
            total_words = len(test_words)
            results[corpus_name] = {
                'top1': correct[1] / total_words,
                'top2': correct[2] / total_words,
                'top3': correct[3] / total_words,
                'precision': precision,
                'recall': recall
            }

            # Write all results to file for this corpus
            results_file = Path(f"./results/{corpus_name}/results_{today}.txt")
            with results_file.open('a') as f:
                f.write("\n\n".join(result_lines) + "\n\n")

        return results
        
def save_results_to_file(results, iteration, folder="results", today=None) -> None:
    # Calculate the date once and reuse it if not provided
    if today is None:
        today = datetime.now().strftime("%Y%m%d")

    # Ensure the folder exists
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"results_iteration_{iteration}_{today}.txt"
    filepath = folder_path / filename
    
    # Construct result content before writing to the file
    result_lines = []
    for corpus_name, accuracies in results.items():
        result_lines.append(f"{corpus_name.capitalize()} Corpus:\n" + "\n".join(
            f"  {acc_type.upper()} Accuracy: {acc_value:.2%}" for acc_type, acc_value in accuracies.items()
        ) + "\n")

    # Write all lines to file in one go
    with open(filepath, 'w') as file:
        file.write("\n".join(result_lines))
    
    print(f"Results saved to {filepath}")

def print_predictions(word: str, predictions: list[tuple[str, float]]) -> None:

    print(f"Word: {word}")
    
    # Calculate total probability of top 3 to normalize and show as percentages
    total_prob = sum(prob for _, prob in predictions)
    if total_prob == 0:
        print("No predictions to display.")
        return

    for rank, (letter, prob) in enumerate(predictions, 1):
        percentage = (prob / total_prob) * 100  # Convert to percentage of the total
        print(f"Rank {rank}: '{letter}' with {percentage:.2f}% of the top 3 confidence")

def main():
    random.seed(42)
    np.random.seed(42)
    iterations = 2
    corpora = ['brown', 'cmu']
    total_accuracy = {corpus_name: {'top1': 0, 'top2': 0, 'top3': 0, 'precision': 0, 'recall': 0} for corpus_name in corpora}

    # Initialize LanguageModel outside the loop
    lm = LanguageModel(q_range=(2, 6))
    lm.load_corpora(use_test_set=False)

    # Generate paths for formatted corpora outside the iteration loop
    formatted_corpus_paths = {
        corpus_name: lm.generate_formatted_corpus(corpus_name, path=f'{corpus_name}_formatted_corpus.txt')
        for corpus_name in corpora
    }

    # Generate and load models outside the iteration loop if they do not change per iteration
    with ThreadPoolExecutor(max_workers=len(corpora)) as executor:
        for corpus_name, corpus_path in formatted_corpus_paths.items():
            executor.submit(lm.generate_and_load_models, corpus_name, corpus_path)

    # Start the iterations
    for iteration in range(iterations):
        lm.prepare_test_set()

        # Evaluate predictions and accumulate accuracies
        accuracies = lm.evaluate_predictions(iteration + 1)
        
        # Print the accuracies for this iteration
        print(f"Iteration {iteration + 1} Accuracies:")
        for corpus_name, corpus_accuracies in accuracies.items():
            print(f"{corpus_name.capitalize()} Corpus:")
            for acc_type, acc_value in corpus_accuracies.items():
                print(f"  {acc_type.upper()} Accuracy: {acc_value:.2%}")
            print()  # Add a newline for spacing
        
        # Accumulate total accuracies
        for corpus_name, accuracy in accuracies.items():
            for acc_type, acc_value in accuracy.items():
                total_accuracy[corpus_name][acc_type] += acc_value

        # Save the results to a file (assuming this is a function that saves to file)
        save_results_to_file(accuracies, iteration + 1)

    # Calculate averaged accuracy
    averaged_accuracy = {corpus_name: {acc_type: acc_value / iterations for acc_type, acc_value in acc_types.items()}
                         for corpus_name, acc_types in total_accuracy.items()}

    # Print final averaged accuracies
    print(f"Averaged accuracy over {iterations} iterations:")
    for corpus_name, acc_types in averaged_accuracy.items():
        print(f"{corpus_name.capitalize()} Corpus:")
        for acc_type, acc_value in acc_types.items():
            print(f"  {acc_type.upper()} Accuracy: {acc_value:.2%}")
        print()

if __name__ == "__main__":
    main()