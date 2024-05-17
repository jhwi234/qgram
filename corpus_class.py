import random
import logging
import regex as reg
from pathlib import Path
import subprocess
from enum import Enum
import nltk
import kenlm

# Enum to define constants for vowels and consonants
class Letters(Enum):
    VOWELS = 'aeèéiîouyæœ'  # Vowel characters
    CONSONANTS = 'bcdfghjklmnpqrstvwxzȝ'  # Consonant characters

    @staticmethod
    def is_vowel(char):
        return char in Letters.VOWELS.value  # Check if character is a vowel

    @staticmethod
    def is_consonant(char):
        return char in Letters.CONSONANTS.value  # Check if character is a consonant

# Function to build language models with KenLM for specified q-gram sizes
def build_kenlm_model(corpus_name, q, corpus_path, model_directory) -> tuple[int, str]:
    arpa_file = model_directory / f"{corpus_name}_{q}gram.arpa"  # ARPA file path
    binary_file = model_directory / f"{corpus_name}_{q}gram.klm"  # Binary file path

    # Build the ARPA model file
    if not run_command(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                       "lmplz failed to generate ARPA model"):
        return q, None

    # Convert the ARPA model to binary format
    if not run_command(['build_binary', '-s', str(arpa_file), str(binary_file)],
                       "build_binary failed to convert ARPA model to binary format"):
        return q, None

    return q, str(binary_file)

# Function to run a command as a subprocess and log any errors
def run_command(command, error_message):
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()}")
        return False

# Class to manage the corpus and handle data processing
class CorpusManager:
    CLEAN_PATTERN = reg.compile(r'\b\p{L}+(?:-\p{L}+)*\b')  # Regex pattern to extract words

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

    # Extract unique characters from the corpus
    def extract_unique_characters(self) -> set:
        return {char for word in self.corpus for char in word}

    # Clean the text by extracting words and filtering based on minimum length
    def clean_text(self, text: str) -> set[str]:
        return {part.lower() for word in self.CLEAN_PATTERN.findall(text) for part in word.split('-') if len(part) >= self.config.min_word_length}

    # Load the corpus from a text file or NLTK corpus
    def load_corpus(self) -> set[str]:
        file_path = self.config.corpus_dir / f'{self.corpus_name}.txt'
        if file_path.is_file():
            with file_path.open('r', encoding='utf-8') as file:
                self.corpus = self.clean_text(file.read())
        else:
            try:
                nltk_corpus_name = self.corpus_name.replace('.txt', '')
                nltk.download(nltk_corpus_name, quiet=True)
                self.corpus = self.clean_text(' '.join(getattr(nltk.corpus, nltk_corpus_name).words()))
            except AttributeError:
                raise ValueError(f"File '{file_path}' does not exist and NLTK corpus '{nltk_corpus_name}' not found.")
            except Exception as e:
                raise RuntimeError(f"Failed to load corpus '{self.corpus_name}': {e}")

        return self.corpus

    # Shuffle and split the corpus into training and test sets
    def _shuffle_and_split_corpus(self) -> tuple[set[str], set[str]]:
        shuffled_corpus = list(self.corpus)
        self.rng.shuffle(shuffled_corpus)
        train_size = int(len(self.corpus) * self.config.split_config)
        return set(shuffled_corpus[:train_size]), set(shuffled_corpus[train_size:])

    # Prepare training and test datasets from the corpus
    def prepare_datasets(self):
        self.train_set, unprocessed_test_set = self._shuffle_and_split_corpus()
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)

        formatted_test_set = []
        for word in unprocessed_test_set:
            num_replacements = min(self.config.num_replacements, len(word))
            modified_word, missing_letters = self._replace_letters(word, num_replacements)
            if missing_letters:
                formatted_test_set.append((modified_word, tuple(missing_letters), word))

        self.test_set = set(formatted_test_set)
        self.all_words = self.train_set.union({original_word for _, _, original_word in self.test_set})

        if self.debug:
            self.save_set_to_file(self.train_set, f'{self.corpus_name}_train_set.txt')
            self.save_set_to_file(self.test_set, f'{self.corpus_name}_formatted_test_set.txt')
            self.save_set_to_file(self.all_words, f'{self.corpus_name}_all_words.txt')

    # Generate a formatted corpus file for KenLM training
    def generate_formatted_corpus(self, data_set, formatted_corpus_path) -> Path:
        formatted_text = '\n'.join(' '.join(word) for word in data_set)
        with formatted_corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_text)
        return formatted_corpus_path

    # Generate and load KenLM models for the corpus
    def generate_models_from_corpus(self, corpus_path):
        model_directory = self.config.model_dir / self.corpus_name
        model_directory.mkdir(parents=True, exist_ok=True)

        model_loaded = False
        for q in self.config.q_range:
            if q not in self.model:
                _, binary_file = build_kenlm_model(self.corpus_name, q, corpus_path, model_directory)
                if binary_file:
                    self.model[q] = kenlm.Model(binary_file)
                    model_loaded = True

        if model_loaded:
            logging.info(f'Model for {q}-gram loaded from {self.corpus_name}')

    # Generate and load models if not already loaded
    def generate_and_load_models(self):
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
        self.generate_models_from_corpus(formatted_train_set_path)

    # Replace letters in the word with underscores
    def _replace_letters(self, word, num_replacements) -> tuple[str, list[str]]:
        modified_word = word
        missing_letters = []
        for _ in range(num_replacements):
            if self.has_replaceable_letter(modified_word):
                modified_word, missing_letter = self._replace_random_letter(modified_word)
                missing_letters.append(missing_letter)
        return modified_word, missing_letters

    # Replace a random letter in the word based on configuration
    def _replace_random_letter(self, word) -> tuple[str, str]:
        vowel_indices = [i for i, letter in enumerate(word) if Letters.is_vowel(letter)]
        consonant_indices = [i for i, letter in enumerate(word) if Letters.is_consonant(letter)]

        if not vowel_indices and not consonant_indices:
            raise ValueError(f"Unable to replace a letter in word: '{word}'.")

        letter_indices = vowel_indices if self.rng.random() < self.config.vowel_replacement_ratio and vowel_indices else consonant_indices or vowel_indices
        letter_index = self.rng.choice(letter_indices)
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        return modified_word, missing_letter
    
    # Check if the word has any replaceable letters
    def has_replaceable_letter(self, word) -> bool:
        return any(Letters.is_vowel(letter) for letter in word) or any(Letters.is_consonant(letter) for letter in word)

    # Save a dataset to a file
    def save_set_to_file(self, data_set, file_name):
        file_path = self.config.sets_dir / file_name
        with file_path.open('w', encoding='utf-8') as file:
            file.writelines(f"{item}\n" for item in data_set)