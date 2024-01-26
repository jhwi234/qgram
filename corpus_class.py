import random
import logging
import regex as reg
from pathlib import Path
import subprocess
from enum import Enum

import nltk
import kenlm

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

class CorpusManager:
    # Regex pattern for extracting words, including hyphenated words, in various scripts.
    # \b indicates word boundaries.
    # \p{L}+ matches one or more Unicode letters, covering a wide range of characters beyond ASCII.
    # (?:-\p{L}+)* allows for optional hyphenated parts, matching additional Unicode letters after a hyphen.
    CLEAN_PATTERN = reg.compile(r'\b\p{L}+(?:-\p{L}+)*\b')

    @staticmethod
    def format_corpus_name(corpus_name) -> str:
        parts = corpus_name.replace('.txt', '').split('_')
        return parts[0] if len(parts) > 1 and parts[0] == parts[1] else corpus_name.replace('.txt', '')

    unique_words_all_corpora = set()  # Static variable to store unique words from all corpora

    @staticmethod
    def add_to_global_corpus(unique_words):
        CorpusManager.unique_words_all_corpora.update(unique_words)

    def __init__(self, corpus_name, config, debug=True):
        self.corpus_name = self.format_corpus_name(corpus_name)
        self.config = config
        self.debug = debug
        self.rng = random.Random(config.seed)
        self.corpus = set()
        self.train_set = set()
        self.test_set = set()
        self.all_words = set()
        self.model = {}
        self.load_corpus()
        self.prepare_datasets()
        self.generate_and_load_models()

    def extract_unique_characters(self) -> set:
        # Use set comprehension for efficiency
        return {char for word in self.corpus for char in word}

    def clean_text(self, text: str) -> set[str]:
        # Extract and clean words from the given text using the defined regex pattern
        # Lowercase each word part and filter by minimum length requirement
        return {part.lower() for word in self.CLEAN_PATTERN.findall(text) for part in word.split('-') if len(part) >= self.config.min_word_length}

    def load_corpus(self) -> set[str]:
        # Check if the corpus is a text file
        file_path = self.config.corpus_dir / f'{self.corpus_name}.txt'
        if file_path.is_file():
            with file_path.open('r', encoding='utf-8') as file:
                # Read the file and clean the text, then store the unique words in self.corpus
                self.corpus = {word for word in self.clean_text(file.read())}
        else:
            # If the corpus is not a file, attempt to load it as an NLTK corpus
            try:
                nltk_corpus_name = self.corpus_name.replace('.txt', '')
                nltk.download(nltk_corpus_name, quiet=True)
                # Retrieve words from the NLTK corpus, clean them, and store in self.corpus
                self.corpus = {word for word in self.clean_text(' '.join(getattr(nltk.corpus, nltk_corpus_name).words()))}
            except AttributeError:
                # This exception is raised if the NLTK corpus does not exist
                raise ValueError(f"File '{file_path}' does not exist and NLTK corpus '{nltk_corpus_name}' not found.")
            except Exception as e:
                # Catch any other unexpected exceptions and provide a more informative error message
                raise RuntimeError(f"Failed to load corpus '{self.corpus_name}': {e}")

        return self.corpus

    def _shuffle_and_split_corpus(self) -> tuple[set[str], set[str]]:
        # Convert the corpus to a list, shuffle it, and then split into training and test sets.
        total_size = len(self.corpus)
        shuffled_corpus = list(self.corpus)
        self.rng.shuffle(shuffled_corpus)  # Randomize the order of the corpus elements
        train_size = int(total_size * self.config.split_config)  # Calculate the size of the training set
        # Split the shuffled corpus into training and test sets and return
        return set(shuffled_corpus[:train_size]), set(shuffled_corpus[train_size:])

    def prepare_datasets(self) -> tuple[set[str], set[str]]:
        # Prepare training and test datasets from the corpus
        self.train_set, unprocessed_test_set = self._shuffle_and_split_corpus()

        # Save the formatted training set for KenLM
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)

        # Process the test set by replacing a letter in each word with an underscore
        formatted_test_set = []
        for word in unprocessed_test_set:
            modified_word, missing_letter, _ = self.replace_random_letter(word)
            formatted_test_set.append((modified_word, missing_letter, word))

        self.test_set = set(formatted_test_set)
        self.all_words = self.train_set.union({original_word for _, _, original_word in self.test_set})

        # Save additional sets in debug mode, including the regular training set
        if self.debug:
            self.save_set_to_file(self.train_set, f'{self.corpus_name}_train_set.txt')
            self.save_set_to_file(self.test_set, f'{self.corpus_name}_formatted_test_set.txt')
            self.save_set_to_file(self.all_words, f'{self.corpus_name}_all_words.txt')

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
                formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
                self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
                self.generate_models_from_corpus(formatted_train_set_path)

    def replace_random_letter(self, word) -> tuple[str, str, str]:
        vowel_indices = [i for i, letter in enumerate(word) if letter in Letters.VOWELS.value]
        consonant_indices = [i for i, letter in enumerate(word) if letter in Letters.CONSONANTS.value]

        if not vowel_indices and not consonant_indices:
            raise ValueError(f"Unable to replace a letter in word: '{word}'.")

        # Prioritize based on the configured ratios
        if self.rng.random() < self.config.vowel_replacement_ratio and vowel_indices:
            letter_indices = vowel_indices
        elif consonant_indices:
            letter_indices = consonant_indices
        else:
            letter_indices = vowel_indices

        letter_index = self.rng.choice(letter_indices)
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        return modified_word, missing_letter, word
    
    def save_set_to_file(self, data_set, file_name):
        file_path = self.config.sets_dir / file_name
        with file_path.open('w', encoding='utf-8') as file:
            file.writelines(f"{item}\n" for item in data_set)
