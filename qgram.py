# Standard library imports
import concurrent.futures
import heapq
import logging
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# Third-party imports
import kenlm
import numpy as np
import nltk
from retry import retry
import csv

logging.basicConfig(level=logging.INFO)
split_pattern = re.compile(r"[-\s]")
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

    def __init__(self, q_range=(1, 7)):
        self.q_range = range(q_range[0], q_range[1] + 1)
        self.models = {}
        self.corpora = {
            'brown': set(),
            'cmu': set(),
            'clmet3': set()
        }
        self.formatted_corpora_cache = {}
        self.test_set = {}
        self.training_corpora = {}
        self.loaded_corpora = False

    def clean_text(self, text: str) -> set[str]:
        # Using regular expressions to remove non-alphabetic characters and split words
        words = clean_pattern.findall(split_pattern.sub(' ', text))
        # Convert to lowercase
        return {word.lower() for word in words}

    def load_text_corpus(self, file_path: str) -> set[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.clean_text(text)

    def load_nltk_corpus(self, corpus_name: str) -> set[str]:
        nltk.download(corpus_name)
        if corpus_name == 'cmudict':
            entries = getattr(nltk.corpus, corpus_name).entries()
            text = ' '.join(word for word, _ in entries)
        else:
            text = ' '.join(getattr(nltk.corpus, corpus_name).words())
        return self.clean_text(text)

    def load_corpora(self, use_test_set: bool = True):
        if not self.loaded_corpora:
            self.download_nltk_resources()
            self.corpora['cmu'] = self.load_nltk_corpus('cmudict')
            self.corpora['brown'] = self.load_nltk_corpus('brown')
            self.corpora['clmet3'] = self.load_text_corpus('CLMET3_words.txt')
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

        return {match.group()
                for word in words if isinstance(word, str)
                for match in clean_pattern.finditer(word.lower())}

    def download_nltk_resources(self):
        resources = ['cmudict', 'brown']
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
        return ''.join([word[:letter_index], '_', word[letter_index+1:]])

    def prepare_test_set(self, n=30000):
        """
        Prepare test and training sets by randomly selecting words from the corpora.
        Either test_set_size or test_set_proportion should be provided. 
        If training_set_size is not provided, it defaults to the remaining words after test set selection.

        :param test_set_size: Absolute number of words in the test set (mutually exclusive with test_set_proportion).
        :param test_set_proportion: Proportion of the corpus to be used as the test set (between 0 and 1).
        :param training_set_size: Absolute number of words in the training set.
        """
        self.test_set = {}
        self.training_corpora = {corpus: set(words) for corpus, words in self.corpora.items()}

        for corpus_name, words in self.corpora.items():
            test_words = set(random.sample(list(words), n))
            self.training_corpora[corpus_name] -= test_words
            self.test_set[corpus_name] = {
                word: self.replace_random_letter(word) for word in test_words
            }

    def generate_formatted_corpus(self, corpus_name, path='formatted_corpus.txt', exclude_test_set=True):
        def format_corpus(words):
            return ' '.join(f'<w> {" ".join(word)} </w>' for word in words)

        words_to_format = self.corpora[corpus_name]
        if exclude_test_set and corpus_name in self.test_set:
            test_words = set(self.test_set[corpus_name].values())
            words_to_format = words_to_format - test_words

        formatted_corpus = format_corpus(words_to_format)
        corpus_path = Path(path)

        with corpus_path.open('w') as f:
            f.write(formatted_corpus)

        return path
    
    def generate_and_load_models(self, corpus_name, corpus_path):
        self.models[corpus_name] = {}
        model_directory = Path(f'{corpus_name}_models')
        model_directory.mkdir(parents=True, exist_ok=True)

        for q in self.q_range:
            _, binary_file = model_task(corpus_name, q, corpus_path, model_directory)
            if binary_file:
                self.models[corpus_name][q] = kenlm.Model(binary_file)
                logging.info(f"Model for {q}-gram loaded for {corpus_name} corpus.")

    def predict_missing_letter(self, corpus_name, oov_word):
            missing_letter_index = oov_word.index('_')
            log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}
            entropy_weights = []
            boundary_start = '<w> ' if missing_letter_index == 0 else ''
            boundary_end = ' </w>' if missing_letter_index == len(oov_word) - 1 else ''
            oov_word_with_boundaries = f"{boundary_start}{oov_word}{boundary_end}"
            for q in self.q_range:
                # q-grams are character grams
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
            # Apply heapq.nlargest to find the top log probabilities
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
        """
        Evaluate predictions for each corpus in the test set.
        """
        today = datetime.now().strftime('%Y%m%d')
        results = {}

        for corpus_name, test_words in self.test_set.items():
            correct = {1: 0, 2: 0, 3: 0}
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            result_lines = []
            test_results = []

            # Create directory before the loop
            directory = Path(f"./results/{corpus_name}")
            directory.mkdir(parents=True, exist_ok=True)

            for original_word, test_word in test_words.items():
                # predictions = self.predict_missing_letter(corpus_name, test_word, self.calculate_letter_probabilities(corpus_name))
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

                result_lines.append(
                    f"Iteration: {iteration_number}\n"
                    f"Test Word: {test_word}\n"
                    f"Correct Word: {original_word}\n"
                    "Top 3 Predictions:\n" +
                    "".join(f"Rank {rank}: '{letter}' with probability {prob:.5f}\n"
                            for rank, (letter, prob) in enumerate(predictions, 1) if rank <= 3)
                )
                test_results.append({
                    'corpus': corpus_name,
                    'test_word': test_word,
                    'correct_word': original_word,
                    'top_predictions': top_predictions,
                    'confidence_scores': [p[1] for p in predictions][:3],
                    'found_at_rank': found_at_rank
                })

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
            
            self.generate_csv_report(corpus_name, test_results)

        return results

    def generate_csv_report(self, corpus_name, test_results):
        today = datetime.now().strftime('%Y%m%d')
        csv_filename = f"./results/{corpus_name}/csv_report_{today}.csv"

        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['Corpus', 'q_range', 'Test Word', 'Correct Word', '1st Top Letter', '1st Confidence Score',
                        '2nd Top Letter', '2nd Confidence Score', '3rd Top Letter', '3rd Confidence Score',
                        'Top Prediction Correct', 'Correct in Top 3']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if file is empty (i.e., at the start of the first iteration)
            if csvfile.tell() == 0:
                writer.writeheader()

            for result in test_results:
                writer.writerow({
                    'Corpus': corpus_name,
                    'q_range': f'{self.q_range.start}-{self.q_range.stop - 1}',
                    'Test Word': result['test_word'],
                    'Correct Word': result['correct_word'],
                    '1st Top Letter': result['top_predictions'][0] if result['top_predictions'] else None,
                    '1st Confidence Score': result['confidence_scores'][0] if result['confidence_scores'] else None,
                    '2nd Top Letter': result['top_predictions'][1] if len(result['top_predictions']) > 1 else None,
                    '2nd Confidence Score': result['confidence_scores'][1] if len(result['confidence_scores']) > 1 else None,
                    '3rd Top Letter': result['top_predictions'][2] if len(result['top_predictions']) > 2 else None,
                    '3rd Confidence Score': result['confidence_scores'][2] if len(result['confidence_scores']) > 2 else None,
                    'Top Prediction Correct': 'Yes' if result['found_at_rank'] == 1 else 'No',
                    'Correct in Top 3': 'Yes' if result['found_at_rank'] is not None else 'No'
                })
    def save_words_to_file(self, words, file_path):
        """
        Save a set of words to a file, one word per line.
        """
        with open(file_path, 'w') as file:
            for word in words:
                file.write(word + '\n')

    def save_training_and_test_words(self, iteration_number):
        for corpus_name in self.corpora:
            training_file_path = Path(f'./results/{corpus_name}_training_words_iteration_{iteration_number}.txt')
            test_file_path = Path(f'./results/{corpus_name}_test_words_iteration_{iteration_number}.txt')

            training_file_path.parent.mkdir(parents=True, exist_ok=True)
            test_file_path.parent.mkdir(parents=True, exist_ok=True)

            if training_file_path.exists():
                print(f"Overwriting existing file for {corpus_name} training data, iteration {iteration_number}")
            self.save_words_to_file(self.training_corpora[corpus_name], training_file_path)

            if test_file_path.exists():
                print(f"Overwriting existing file for {corpus_name} test data, iteration {iteration_number}")
            self.save_words_to_file(set(self.test_set[corpus_name].values()), test_file_path)

            print(f"Saved training and test words for {corpus_name} corpus, iteration {iteration_number}")

def save_results_to_file(results, iteration, folder="results"):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    filename = f"results_iteration_{iteration}_{today}.txt"
    filepath = folder_path / filename
    
    lines = []
    for corpus_name, accuracies in results.items():
        lines.append(f"{corpus_name.capitalize()} Corpus:\n")
        lines.extend(f"  {acc_type.upper()} Accuracy: {acc_value:.2%}\n" for acc_type, acc_value in accuracies.items())
        lines.append("\n")
    
    content = ''.join(lines)
    filepath.write_text(content)
    
    print(f"Results saved to {filepath}")

def print_predictions(word: str, predictions: list[tuple[str, float]]):
    logger = logging.getLogger(__name__)

    logger.info(f"Word: {word}")
    
    # Calculate total probability of top 3 to normalize and show as percentages
    total_prob = sum(prob for _, prob in predictions)
    if total_prob == 0:
        logger.info("No predictions to display.")
        return

    for rank, (letter, prob) in enumerate(predictions, 1):
        percentage = (prob / total_prob) * 100
        logger.info(f"Rank {rank}: '{letter}' with {percentage:.2f}% of the top 3 confidence")

def print_accuracies(accuracies, prefix=""):
    print(f"{prefix}: ")
    for corpus_name, corpus_accuracies in accuracies.items():
        print(f"{corpus_name.capitalize()} Corpus:")
        for acc_type, acc_value in corpus_accuracies.items():
            print(f"  {acc_type.upper()} Accuracy: {acc_value:.2%}")
        print()

def accumulate_accuracies(source, target):
    for corpus_name, accuracy in source.items():
        for acc_type, acc_value in accuracy.items():
            target[corpus_name][acc_type] += acc_value

def calculate_average_accuracies(total_accuracy, iterations):
    return {
        corpus_name: {
            acc_type: acc_value / iterations
            for acc_type, acc_value in acc_types.items()
        }
        for corpus_name, acc_types in total_accuracy.items()
    }

def check_data_leakage(training_corpora, test_set):
    for corpus_name, test_words in test_set.items():
        training_words = training_corpora[corpus_name]
        for test_word in test_words.values():
            assert test_word not in training_words, f"Data leakage detected in {corpus_name}: {test_word} found in training data"

def main_iteration(lm, corpora, total_accuracy, iteration):
    lm.prepare_test_set()
    check_data_leakage(lm.training_corpora, lm.test_set)
    lm.save_training_and_test_words(iteration)

    formatted_corpus_paths = {corpus_name: lm.generate_formatted_corpus(corpus_name, path=f'{corpus_name}_formatted_corpus.txt') for corpus_name in corpora}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(lm.generate_and_load_models, corpus_name, corpus_path): corpus_name for corpus_name, corpus_path in formatted_corpus_paths.items()}

        for future in concurrent.futures.as_completed(futures):
            corpus_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"{corpus_name} generated an exception: {exc}")

    accuracies = lm.evaluate_predictions(iteration)
    print_accuracies(accuracies, f"Iteration {iteration}")
    accumulate_accuracies(accuracies, total_accuracy)
    save_results_to_file(accuracies, iteration)

def main():
    random.seed(42)
    np.random.seed(42)
    iterations = 10
    corpora = ['brown', 'cmu', 'clmet3']

    total_accuracy = {corpus_name: {'top1': 0, 'top2': 0, 'top3': 0, 'precision': 0, 'recall': 0} for corpus_name in corpora}
    lm = LanguageModel(q_range=(6, 6))
    lm.load_corpora(use_test_set=False)

    formatted_corpus_paths = {
        corpus_name: lm.generate_formatted_corpus(corpus_name, path=f'{corpus_name}_formatted_corpus.txt')
        for corpus_name in corpora
    }

    for iteration in range(1, iterations + 1):
        main_iteration(lm, formatted_corpus_paths, total_accuracy, iteration)

    averaged_accuracy = calculate_average_accuracies(total_accuracy, iterations)
    print_accuracies(averaged_accuracy, f"Averaged accuracy over {iterations} iterations")

if __name__ == "__main__":
    main()
