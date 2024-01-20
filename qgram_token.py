### this version of the script generates word lists that maintain the original number of the tokens in the training list rather than turning it into a list of unique tokens

import random
import logging
import regex as reg
import csv
from pathlib import Path
import subprocess
from enum import Enum
from collections import Counter

import nltk
import kenlm
from evaluation_class import EvaluateModel

# Define constants for vowels and consonants using Enum for better organizationclass Letters(Enum):
class Letters(Enum):
    VOWELS = 'aeiouæœ'
    CONSONANTS = 'bcdfghjklmnpqrstvwxyz'

    @staticmethod
    def is_vowel(char):
        return char in Letters.VOWELS.value

    @staticmethod
    def is_consonant(char):
        return char in Letters.CONSONANTS.value

# Function to build language models with KenLM for specified q-gram sizes
def build_kenlm_model(corpus_name, q, corpus_path, model_directory) -> tuple[int, str]:
    arpa_file = model_directory / f"{corpus_name}_{q}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q}gram.klm"

    try:
        # Build ARPA file and convert it to binary format for efficient usage
        with subprocess.Popen(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.SubprocessError(f"lmplz failed: {stderr.decode()}")

        with subprocess.Popen(['build_binary', '-s', str(arpa_file), str(binary_file)], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.SubprocessError(f"build_binary failed: {stderr.decode()}")

    except subprocess.SubprocessError as e:
        logging.error(f"Error in building KenLM model for {corpus_name} with q={q}: {e}")
        return q, None

    # Return q-gram size and path to the binary model file
    return q, str(binary_file)

# Configuration class for language model testing parameters. Change the testing inputs here.class Config:
class Config:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir else __file__).parent
        self.data_dir = self.base_dir / 'data'
        self.model_dir = self.data_dir / 'models'
        self.log_dir = self.data_dir / 'logs'
        self.corpus_dir = self.data_dir / 'corpora'
        self.output_dir = self.data_dir / 'outputs'
        self.text_dir = self.output_dir / 'texts'
        self.csv_dir = self.output_dir / 'csv'
        self.sets_dir = self.output_dir / 'sets'
        
        # Default values for other configurations
        self.seed = 42
        self.q_range = [6, 6]
        self.split_config = 0.5
        self.vowel_replacement_ratio = 0.2
        self.consonant_replacement_ratio = 0.8
        self.min_word_length = 3
        self.prediction_method_name = 'context_sensitive'
        self.log_level = logging.INFO

    # Logging Configuration: Setup log file and console output formats
    def setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logfile = self.log_dir / 'logfile.log'
        file_handler = logging.FileHandler(logfile, mode='a')
        file_format = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_format)

        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)

        logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    def create_directories(self):
        for directory in [self.data_dir, self.model_dir, self.log_dir, self.corpus_dir, self.output_dir, self.sets_dir, self.text_dir, self.csv_dir]:
            directory.mkdir(exist_ok=True)

class CorpusManager:
    CLEAN_PATTERN = reg.compile(r'\b\p{L}+(?:-\p{L}+)*\b')

    corpora_tokens = []  # List to store all words across corpora

    @staticmethod
    def add_to_global_corpus(words):
        """Add words to the global list of words across all corpora."""
        CorpusManager.corpora_tokens.extend(words)

    @staticmethod
    def format_corpus_name(corpus_name) -> str:
        parts = corpus_name.replace('.txt', '').split('_')
        return parts[0] if len(parts) > 1 and parts[0] == parts[1] else corpus_name.replace('.txt', '')

    def __init__(self, corpus_name, config, split_type='A', debug=False):
        self.corpus_name = self.format_corpus_name(corpus_name)
        self.config = config
        self.split_type = split_type
        self.debug = debug
        self.rng = random.Random(config.seed)
        self.corpus = Counter()
        self.load_corpus()
        self.train_set = [] # Not a set in the python sense allows for duplicates
        self.test_set = [] # Not a set in the python sense allows for duplicates
        self.model = {}
        self.all_words = set()
        self.prepare_datasets()

    def extract_unique_characters(self) -> set:
        return {char for word in self.corpus for char in word}

    def clean_text(self, text: str) -> list[str]:
        return [part.lower() for word in self.CLEAN_PATTERN.findall(text) for part in word.split('-') if len(part) >= self.config.min_word_length]

    def load_corpus(self):
        file_path = self.config.corpus_dir / f'{self.corpus_name}.txt'
        if file_path.is_file():
            with file_path.open('r', encoding='utf-8') as file:
                for line in file:
                    self.corpus.update(self.clean_text(line))
        else:
            try:
                nltk_corpus_name = self.corpus_name.replace('.txt', '')
                nltk.download(nltk_corpus_name, quiet=True)
                self.corpus.update(self.clean_text(' '.join(getattr(nltk.corpus, nltk_corpus_name).words())))
            except AttributeError:
                raise ValueError(f"File '{file_path}' does not exist and NLTK corpus '{nltk_corpus_name}' not found.")
            except Exception as e:
                raise RuntimeError(f"Failed to load corpus '{self.corpus_name}': {e}")

    def prepare_datasets(self):
        """
        Prepares training and testing datasets based on the chosen split type (A or B).
        Split Type A segregates unique word types, while Type B maintains original word frequencies.
        Updates the all_words set for comprehensive evaluation checks.

        Args:
            split_type (str): 'A' for splitting based on unique words, 'B' for maintaining word frequency.
        """
        # Choose the method to split the corpus based on the specified split type
        if self.split_type == 'A':
            self._split_type_a()
        else:
            self._split_type_b()

        # Update all_words set
        self.all_words = set(self.train_set)
        self.all_words.update([original_word for _, _, original_word in self.test_set])

        # Generate the formatted training list path and the corresponding models
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
        self.generate_models_from_corpus(formatted_train_set_path)

    def _split_type_a(self):
        """
        Split Type A: Divides the list of unique words into training and testing sets based on the split configuration.
        The token frequencies associated with each word are included in the respective sets.
        """
        # Extract unique word types and shuffle them
        unique_word_types = list(self.corpus.keys())
        self.rng.shuffle(unique_word_types)

        # Calculate the split index based on the split configuration
        split_index = int(len(unique_word_types) * self.config.split_config)

        # Divide the shuffled words into training and testing sets
        training_word_types = unique_word_types[:split_index]
        testing_word_types = unique_word_types[split_index:]

        # Initialize a set to track used word-letter combinations in the test set
        used_combinations = set()

        # Assign token frequencies to the training set
        for word in training_word_types:
            self.train_set.extend([word] * self.corpus[word])

        # Assign token frequencies to the testing set, with modified words
        for word in testing_word_types:
            word_count = self.corpus[word]
            for _ in range(word_count):
                modified_word, missing_letter, _ = self.replace_random_letter(word, used_combinations)
                self.test_set.append((modified_word, missing_letter, word))
                used_combinations.add((word, missing_letter))

        # Save lists to files if debug mode is active
        if self.debug:
            self.save_list_to_file(self.train_set, f'{self.corpus_name}_train_list_a.txt')
            self.save_list_to_file(self.test_set, f'{self.corpus_name}_test_list_a.txt')

    def _split_type_b(self):
        """
        Split Type B: Randomly shuffles and divides the entire corpus into training and test lists.
        Allows the same word tokens to appear in both training and testing data, maintaining word frequency balance.
        Ensures no duplicate word-letter combinations in the test set.
        """
        # Flatten the corpus into a list of word tokens
        all_word_tokens = []
        for word, count in self.corpus.items():
            all_word_tokens.extend([word] * count)
        self.rng.shuffle(all_word_tokens)

        # Determine the split point for training and test sets
        split_index = int(len(all_word_tokens) * self.config.split_config)
        train_tokens = all_word_tokens[:split_index]
        test_tokens = all_word_tokens[split_index:]

        # Assign tokens to the training list
        self.train_set.extend(train_tokens)

        # Initialize a set to track used word-letter combinations in test set
        used_combinations = set()
        for word in test_tokens:
            modified_word, missing_letter, _ = self.replace_random_letter(word, used_combinations)
            self.test_set.append((modified_word, missing_letter, word))
            used_combinations.add((word, missing_letter))

        # Save lists to files if debug mode is active
        if self.debug:
            self.save_list_to_file(self.train_set, f'{self.corpus_name}_train_list_b.txt')
            self.save_list_to_file(self.test_set, f'{self.corpus_name}_test_list_b.txt')

    def generate_formatted_corpus(self, data_set, formatted_corpus_path) -> Path:
        # Prepare a corpus file formatted for KenLM training, with each word on a new line
        formatted_text = [' '.join(word) for word in data_set]
        formatted_corpus = '\n'.join(formatted_text)

        # Save the formatted corpus to a file
        with formatted_corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_corpus)

        return formatted_corpus_path

    def generate_models_from_corpus(self, corpus_path):
        # Create the directory for storing language models
        model_directory = self.config.model_dir / self.corpus_name
        model_directory.mkdir(parents=True, exist_ok=True)

        model_loaded = False
        for q in self.config.q_range:
            if q not in self.model:
                # Generate and load KenLM models for each q-gram size
                _, binary_file = build_kenlm_model(self.corpus_name, q, corpus_path, model_directory)
                if binary_file:
                    self.model[q] = kenlm.Model(binary_file)
                    model_loaded = True

        if model_loaded:
            logging.info(f'Model for {q}-gram loaded from {self.corpus_name}')

    def generate_and_load_models(self):
        # Generate and load models only if they haven't been loaded for the specified q-range
        for q in self.config.q_range:
            if q not in self.model:
                formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_list.txt'
                self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
                self.generate_models_from_corpus(formatted_train_set_path)

    def replace_random_letter(self, word, used_combinations) -> tuple[str, str, str]:
        vowel_indices = [i for i, letter in enumerate(word) if letter in Letters.VOWELS.value]
        consonant_indices = [i for i, letter in enumerate(word) if letter in Letters.CONSONANTS.value]

        if not vowel_indices and not consonant_indices:
            raise ValueError(f"Unable to replace a letter in word: '{word}'.")

        # Filter indices to only those not used before
        valid_vowel_indices = [i for i in vowel_indices if (word, i) not in used_combinations]
        valid_consonant_indices = [i for i in consonant_indices if (word, i) not in used_combinations]

        # Choose from the valid indices
        letter_indices = valid_vowel_indices if self.rng.random() < self.config.vowel_replacement_ratio and valid_vowel_indices else valid_consonant_indices
        if not letter_indices:
            letter_indices = valid_vowel_indices or vowel_indices  # Fallback if no valid consonant indices

        letter_index = self.rng.choice(letter_indices)
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        return modified_word, missing_letter, word

    def save_list_to_file(self, data_list, file_name):
        file_path = self.config.sets_dir / file_name

        # Aggregate the data into a single string.
        aggregated_data = []
        for item in data_list:
            if isinstance(item, tuple):
                # Formatting tuple as ('word', 'letter', 'original_word')
                formatted_item = f"('{item[0]}', '{item[1]}', '{item[2]}')"
                aggregated_data.append(formatted_item)
            else:
                # If it's not a tuple, just write the item followed by a new line
                aggregated_data.append(item)
        
        # Join all items into a single string with new lines.
        aggregated_data_str = '\n'.join(aggregated_data)

        # Write the aggregated string to the file in one go.
        with file_path.open('w', encoding='utf-8', buffering=8192) as file:  # 8192 bytes buffer size
            file.write(aggregated_data_str)

def run(corpus_name, config, split_type):
    # Use the static method from CorpusManager to format the corpus name
    formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
    logging.info(f'Processing {formatted_corpus_name} Corpus with split type {split_type}')
    logging.info('-' * 40)

    # Initialize CorpusManager with the formatted corpus name, configuration settings, and split type
    corpus_manager = CorpusManager(formatted_corpus_name, config, split_type)

    # Add unique words from the current corpus to the global corpus
    CorpusManager.add_to_global_corpus(corpus_manager.corpus)

    # Create an EvaluateModel instance, passing the CorpusManager instance
    eval_model = EvaluateModel(corpus_manager)

    # Retrieve the prediction method based on the configuration
    prediction_method = getattr(eval_model.predictor, config.prediction_method_name)

    # Evaluate character predictions using the selected prediction method
    evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)

    # Log the results of the evaluation for accuracy and validity
    logging.info(f'Evaluated with: {prediction_method.__name__}')
    logging.info(f'Model evaluation completed for: {corpus_name}')
    for i in range(1, 4):
        logging.info(f'TOP{i} ACCURACY: {evaluation_metrics["accuracy"][i]:.2%} | TOP{i} VALIDITY: {evaluation_metrics["validity"][i]:.2%}')

    # Export the prediction details and summary statistics to CSV and text files
    eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
    eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)

    # Save recall and precision statistics
    eval_model.save_recall_precision_stats(evaluation_metrics)

    logging.info('-' * 40)

def main():
    config = Config()
    config.setup_logging()
    config.create_directories()

    # Process each corpus with both split types A and B
    corpora = ['brown', 'cmudict', 'CLMET3.txt', 'reuters', 'gutenberg', 'inaugural']
    for corpus_name in corpora:
        for split_type in ['A', 'B']:
            run(corpus_name, config, split_type)

    # Create and evaluate the mega-corpus
    mega_corpus_name = 'mega_corpus'
    with open(config.corpus_dir / f'{mega_corpus_name}.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(CorpusManager.corpora_tokens))
    run(mega_corpus_name, config, 'A')  # You can change to 'B' or loop for both types

if __name__ == '__main__':
    main()