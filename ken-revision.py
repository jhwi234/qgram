import random
import logging
import re
from pathlib import Path
import enum
import numpy as np

import kenlm
from prediction_methods import Predictions
import nltk
import subprocess
from retry import retry

# Constants
VOWELS = 'aeiouæœ'
CONSONANTS = ''.join(set('abcdefghijklmnopqrstuvwxyz') - set(VOWELS))
seed = 42

# Suppress NLTK download messages
nltk.download('punkt', quiet=True)

# Enumeration for letter types
class LetterType(enum.Enum):
    VOWEL = 1
    CONSONANT = 2

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
    def __init__(self, corpus_name, q_range=(6, 6), split_config=0.9, seed=seed):
        self.corpus_name = corpus_name
        self.q_range = range(q_range[0], q_range[1] + 1)
        self.seed = seed
        self.model = {}
        self.predictor = Predictions(self.model, self.q_range) 
        self.corpus = set()
        self.test_set = set()
        self.training_set = set()
        self.all_words = set()
        self.rng = random.Random(seed)
        self.split_config = split_config

    def load_corpus(self, corpus_source, is_file=False):
        if is_file:
            with open(corpus_source, 'r', encoding='utf-8') as file:
                text_data = file.read()
        else:
            text_data = ' '.join(corpus_source())

        tokenized_data = nltk.word_tokenize(text_data)
        self.corpus = self.clean_text(' '.join(tokenized_data))

    def clean_text(self, text: str) -> set[str]:
        clean_pattern = re.compile(r'\b[a-zA-Z]{4,}\b')
        return set(word.lower() for word in clean_pattern.findall(text))

    def prepare_datasets(self, vowel_replacement_ratio=0.3, consonant_replacement_ratio=0.7):
        total_size = len(self.corpus)
        shuffled_corpus = list(self.corpus)
        training_size = int(total_size * self.split_config)
        self.rng.shuffle(shuffled_corpus)

        self.training_set = set(shuffled_corpus[:training_size])
        unprocessed_test_set = set(shuffled_corpus[training_size:])

        self.save_set_to_file(self.training_set, f"{self.corpus_name}_training_set.txt")

        self.test_set = {self.replace_random_letter(word, include_original=True, vowel_replacement_ratio=vowel_replacement_ratio, consonant_replacement_ratio=consonant_replacement_ratio) for word in unprocessed_test_set}
        self.save_set_to_file(self.test_set, f"{self.corpus_name}_formatted_test_set.txt")

        self.all_words = self.training_set.union(self.test_set)

    def generate_and_load_models(self):
        training_corpus_path = self.generate_formatted_corpus(self.training_set, f"{self.corpus_name}_formatted_training_set.txt")
        self.generate_models_from_corpus(training_corpus_path)

    def generate_formatted_corpus(self, data_set, path):
        formatted_text = [" ".join(word) for word in data_set]
        formatted_corpus = '\n'.join(formatted_text)
        corpus_path = Path(path)
        with corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_corpus)
        return path

    def generate_models_from_corpus(self, corpus_path):
        model_directory = Path(f'{self.corpus_name}_models')
        model_directory.mkdir(parents=True, exist_ok=True)

        for q in self.q_range:
            _, binary_file = model_task(self.corpus_name, q, corpus_path, model_directory)
            if binary_file:
                self.model[q] = kenlm.Model(binary_file)
                logging.info(f"Model for {q}-gram loaded.")

    def prepare_and_save_test_set(self, data_set, file_name):
        formatted_test_set = []
        for word in data_set:
            modified_word, missing_letter = self.replace_random_letter(word)
            # Append a tuple containing the modified word, the missing letter, and the original word
            formatted_test_set.append((modified_word, missing_letter, word))

        self.save_set_to_file(formatted_test_set, file_name)

    def replace_random_letter(self, word, include_original=False, vowel_replacement_ratio=0.5, consonant_replacement_ratio=0.5):
        # Categorize indices of vowels and consonants in the word.
        categorized_indices = {
            LetterType.VOWEL: [i for i, letter in enumerate(word) if letter in VOWELS],
            LetterType.CONSONANT: [i for i, letter in enumerate(word) if letter in CONSONANTS]
        }

        # Randomly decide whether to replace a vowel and/or a consonant.
        replace_vowel = self.rng.random() < vowel_replacement_ratio and categorized_indices[LetterType.VOWEL]
        replace_consonant = self.rng.random() < consonant_replacement_ratio and categorized_indices[LetterType.CONSONANT]

        # Select the type of letter to replace.
        if replace_vowel and replace_consonant:
            letter_type = self.rng.choice([LetterType.VOWEL, LetterType.CONSONANT])
        elif replace_vowel:
            letter_type = LetterType.VOWEL
        elif replace_consonant:
            letter_type = LetterType.CONSONANT
        else:
            # Fallback if no letter is selected for replacement.
            letter_type = self.rng.choice([LetterType.VOWEL, LetterType.CONSONANT])
            if not categorized_indices[letter_type]:
                # Fallback to any letter if the chosen type is not in the word.
                letter_type = LetterType.VOWEL if letter_type == LetterType.CONSONANT else LetterType.CONSONANT

        # Choose a random index from the selected letter type for replacement.
        letter_indices = categorized_indices[letter_type]
        letter_index = self.rng.choice(letter_indices)

        # Replace the selected letter with a placeholder and store the missing letter.
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        # Return the results.
        if include_original:
            return (modified_word, missing_letter, word)
        else:
            return modified_word, missing_letter

    def evaluate_model(self, output_file, prediction_method):
        correct_counts = {1: 0, 2: 0, 3: 0}
        total_words = len(self.test_set)
        top1_valid_predictions = 0
        top2_valid_predictions = 0
        top3_valid_predictions = 0
        predictions = []

        for test_data in self.test_set:
            modified_word, missing_letter, original_word = test_data
            top_three_predictions = prediction_method(modified_word) 

            correct_found = False
            for rank, (predicted_letter, _) in enumerate(top_three_predictions, start=1):
                reconstructed_word = modified_word.replace('_', predicted_letter)
                if predicted_letter == missing_letter:
                    correct_found = True
                    for i in range(rank, 4):
                        correct_counts[i] += 1  # Update correct counts
                    if rank == 1:
                        top1_valid_predictions += 1
                    if rank <= 2:
                        top2_valid_predictions += 1
                    top3_valid_predictions += 1
                    break  # Stop checking if correct letter is found

                # Check for valid word in TOP1 and TOP2 predictions
                if rank == 1 and reconstructed_word in self.all_words:
                    top1_valid_predictions += 1
                if rank == 2 and reconstructed_word in self.all_words:
                    top2_valid_predictions += 1

            # Check for any valid word in TOP3 predictions if correct letter wasn't found
            if not correct_found:
                if any(modified_word.replace('_', pred_letter) in self.all_words for pred_letter, _ in top_three_predictions):
                    top3_valid_predictions += 1

            predictions.append((modified_word, missing_letter, top_three_predictions))

        top1_recall = top1_valid_predictions / total_words if total_words > 0 else 0.0
        top2_recall = top2_valid_predictions / total_words if total_words > 0 else 0.0
        top3_recall = top3_valid_predictions / total_words if total_words > 0 else 0.0

        self.save_predictions_to_file(correct_counts, top1_recall, top2_recall, top3_recall, total_words, predictions, output_file)

        return correct_counts, top1_recall, top2_recall, top3_recall

    def save_predictions_to_file(self, correct_counts, top1_recall, top2_recall, top3_recall, total_words, predictions, file_name):
        with open(file_name, 'w') as file:
            # Write accuracies and metrics
            file.write(f"TOP1 PRECISION: {correct_counts[1] / total_words:.2%}\n")
            file.write(f"TOP2 PRECISION: {correct_counts[2] / total_words:.2%}\n")
            file.write(f"TOP3 PRECISION: {correct_counts[3] / total_words:.2%}\n")
            file.write(f"TOP1 RECALL: {top1_recall:.2%}\n")
            file.write(f"TOP2 RECALL: {top2_recall:.2%}\n")
            file.write(f"TOP3 RECALL: {top3_recall:.2%}\n\n")

            # Write predictions for each word
            for modified_word, missing_letter, top_three_predictions in predictions:
                top_predicted_letter, top_confidence = top_three_predictions[0]
                file.write(f"Test Word: {modified_word}\nOriginal: {missing_letter}\n")
                file.write(f"Predicted: {top_predicted_letter}\n")
                file.write("Top Three Predictions:\n")
                for rank, (letter, confidence) in enumerate(top_three_predictions, start=1):
                    file.write(f"Rank {rank}: '{letter}' confidence {confidence:.7f}\n")
                file.write("\n")

    def save_set_to_file(self, data_set, file_name):
        with open(file_name, 'w') as file:
            for item in data_set:
                formatted_line = self.format_data_for_saving(item)
                file.write(formatted_line + "\n")

    @staticmethod
    def format_data_for_saving(item):
        if isinstance(item, tuple):
            return f"({item[0]}, {item[1]}, {item[2]})"
        else:
            return f"{item}"

def load_nltk_corpus(corpus):
    nltk.download(corpus, quiet=True)
    return getattr(nltk.corpus, corpus).words()

def load_file_corpus(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read().split()

# Set up logging
def setup_logging():
    # Create a logging handler that writes log messages to a file
    file_handler = logging.FileHandler('logfile.log', mode='a')
    file_format = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Create a logging handler that outputs minimal information to the console
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # Set up the root logger with both handlers
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def process_corpus(corpus_name, corpus_loader):
    print(f"Processing {corpus_name} Corpus")
    print("-" * 40)

    lm = LanguageModel(corpus_name)
    lm.load_corpus(corpus_loader)
    logging.info(f"{corpus_name} Language model initialized")

    # Accessing default parameters for vowel and consonant replacement ratios
    vowel_ratio, consonant_ratio = LanguageModel.prepare_datasets.__defaults__[-2:]
    logging.info(f"Vowel Replacement Ratio: {vowel_ratio}")
    logging.info(f"Consonant Replacement Ratio: {consonant_ratio}")

    lm.prepare_datasets()
    logging.info(f"Training set size: {len(lm.training_set)}")
    logging.info(f"Test set size: {len(lm.test_set)}")

    lm.generate_and_load_models()
    logging.info(f"{corpus_name} Q-gram models generated and loaded")

    prediction_method = lm.predictor.entropy_weighted_prediction
    logging.info(f"Evaluated with: {prediction_method.__name__}")

    correct_counts, top1_recall, top2_recall, top3_recall = lm.evaluate_model(f"{corpus_name}_predictions.txt", prediction_method)
    logging.info(f"Model evaluation completed for: {corpus_name}")
    logging.info(f"TOP1 PRECISION: {correct_counts[1] / len(lm.test_set):.2%}")
    logging.info(f"TOP2 PRECISION: {correct_counts[2] / len(lm.test_set):.2%}")
    logging.info(f"TOP3 PRECISION: {correct_counts[3] / len(lm.test_set):.2%}")
    logging.info(f"TOP1 RECALL: {top1_recall:.2%}")
    logging.info(f"TOP2 RECALL: {top2_recall:.2%}")
    logging.info(f"TOP3 RECALL: {top3_recall:.2%}")
    print("-" * 40)

def main():
    setup_logging()
    process_corpus("cmudict", lambda: load_nltk_corpus('cmudict'))
    process_corpus("brown", lambda: load_nltk_corpus('brown'))
    process_corpus("CLMET3", lambda: load_file_corpus("CLMET3_words.txt"))

if __name__ == "__main__":
    main()