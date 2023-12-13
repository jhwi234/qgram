import random
import logging
import re
import csv
from pathlib import Path
import subprocess
from collections import defaultdict
from enum import Enum

import nltk
import kenlm
from prediction_methods import Predictions

# Directory Setup: Define paths for data and output directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
CORPUS_DIR = DATA_DIR / "corpora"
OUTPUT_DIR = DATA_DIR / "outputs"
TEXT_DIR = OUTPUT_DIR / "texts"
CSV_DIR = OUTPUT_DIR / "csv"
SETS_DIR = OUTPUT_DIR / "sets"

# Ensures necessary directories exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, CORPUS_DIR, OUTPUT_DIR, SETS_DIR, TEXT_DIR, CSV_DIR]:
    directory.mkdir(exist_ok=True)

# Logging Configuration: Setup log file and console output formats
def setup_logging():
    logfile = LOG_DIR / 'logfile.log'
    file_handler = logging.FileHandler(logfile, mode='a')
    file_format = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# Define constants for vowels and consonants using Enum for better organization
class Letters(Enum):
    VOWELS = 'aeiouæœ'
    CONSONANTS = ''.join(set('abcdefghijklmnopqrstuvwxyzæœ') - set(VOWELS))

# Function to build language models with KenLM for specified q-gram sizes
def run_kenlm(corpus_name, q, corpus_path, model_directory) -> tuple[int, str]:
    arpa_file = model_directory / f'{corpus_name}_{q}gram.arpa'
    binary_file = model_directory / f'{corpus_name}_{q}gram.klm'

    # Build ARPA file and convert it to binary format for efficient usage
    subprocess.run(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['build_binary', '-s', str(arpa_file), str(binary_file)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Return q-gram size and path to the binary model file
    return q, str(binary_file)

# Configuration class for language model testing parameters
class Config:
    def __init__(self, seed: int = 42, q_range: tuple = (6, 6), split_config: float = 0.5,
                 vowel_replacement_ratio: float = 0.5, consonant_replacement_ratio: float = 0.5, 
                 min_word_length: int = 4):
        self.seed = seed
        self.q_range = q_range
        self.split_config = split_config
        self.vowel_replacement_ratio = vowel_replacement_ratio
        self.consonant_replacement_ratio = consonant_replacement_ratio
        self.min_word_length = min_word_length

# Language model class for processing and predicting text
class LanguageModel:
    # Regex pattern to extract words, allowing for hyphenated words.
    # \b indicates word boundary, [a-zA-Z]+ matches one or more letters, and (?:-[a-zA-Z]+)* allows for optional hyphenated parts.
    CLEAN_PATTERN = re.compile(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b')

    def __init__(self, corpus_name, config):
        # Remove '.txt' extension from corpus name if present
        self.corpus_name = corpus_name.replace('.txt', '')
        # Store configuration parameters
        self.config = config
        # Define range of q-grams to be used in the model
        self.q_range = range(config.q_range[0], config.q_range[1] + 1)
        # Initialize dictionary to store models for each q-gram
        self.model = {}
        # Initialize prediction class with models and q-gram range
        self.predictor = Predictions(self.model, self.q_range)
        # Initialize sets for storing corpus, test set, and training set
        self.corpus = set()
        self.test_set = set()
        self.training_set = set()
        # Initialize random number generator with seed for reproducibility
        self.rng = random.Random(config.seed)
        # Initialize set to store all words (union of training and test sets)
        self.all_words = set()
        # Store other configuration parameters
        self.split_config = config.split_config
        self.vowel_replacement_ratio = config.vowel_replacement_ratio
        self.consonant_replacement_ratio = config.consonant_replacement_ratio
        self.min_word_length = config.min_word_length

        # Log initialization information
        logging.info(f"Language Model for {self.corpus_name} initialized with:")
        logging.info(f"Seed: {config.seed}")
        logging.info(f"Q-gram Range: {config.q_range}")
        logging.info(f"Train-Test Split Configuration: {self.split_config}")
        logging.info(f"Vowel Replacement Ratio: {self.vowel_replacement_ratio}")
        logging.info(f"Consonant Replacement Ratio: {self.consonant_replacement_ratio}")
        logging.info(f"Minimum Word Length: {self.min_word_length}")

    def clean_text(self, text: str) -> set[str]:
        # Extract and clean words from the given text using the defined regex pattern
        # Lowercase each word part and filter by minimum length requirement
        return {part.lower() for word in self.CLEAN_PATTERN.findall(text) for part in word.split('-') if len(part) >= self.min_word_length}

    def load_corpus(self, corpus_name) -> set[str]:
        # Load corpus data from a text file or an NLTK corpus
        if corpus_name.endswith('.txt'):
            # If corpus is a file, read it and extract words
            file_path = CORPUS_DIR / corpus_name
            with file_path.open('r', encoding='utf-8') as file:
                self.corpus = {word for word in self.clean_text(file.read())}
        else:
            # If corpus is an NLTK corpus, download and extract words
            nltk.download(corpus_name, quiet=True)
            self.corpus = {word for word in self.clean_text(' '.join(getattr(nltk.corpus, corpus_name).words()))}

    def _shuffle_and_split_corpus(self) -> tuple[set[str], set[str]]:
        # Convert the corpus to a list, shuffle it, and then split into training and test sets.
        total_size = len(self.corpus)
        shuffled_corpus = list(self.corpus)
        self.rng.shuffle(shuffled_corpus)  # Randomize the order of the corpus elements
        training_size = int(total_size * self.split_config)  # Calculate the size of the training set
        # Split the shuffled corpus into training and test sets and return
        return set(shuffled_corpus[:training_size]), set(shuffled_corpus[training_size:])

    def prepare_datasets(self):
        # Prepare training and test datasets from the corpus
        self.training_set, unprocessed_test_set = self._shuffle_and_split_corpus()
        self.save_set_to_file(self.training_set, f"{self.corpus_name}_training_set.txt")

        # Process the test set by replacing a letter in each word with an underscore
        formatted_test_set = []
        for word in unprocessed_test_set:
            modified_word, missing_letter = self.replace_random_letter(word)
            formatted_test_set.append((modified_word, missing_letter, word))

        self.test_set = set(formatted_test_set)
        self.save_set_to_file(self.test_set, f"{self.corpus_name}_formatted_test_set.txt")
        # Combine training and test sets for a comprehensive word list
        self.all_words = self.training_set.union({original_word for _, _, original_word in self.test_set})
        self.save_set_to_file(self.all_words, f"{self.corpus_name}_all_words.txt")

    def generate_formatted_corpus(self, data_set, formatted_corpus_path) -> Path:
        # Prepare a corpus file formatted for KenLM training, with each word on a new line
        formatted_text = [" ".join(word) for word in data_set]
        formatted_corpus = '\n'.join(formatted_text)

        # Save the formatted corpus to a file
        with formatted_corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_corpus)

        return formatted_corpus_path

    def generate_models_from_corpus(self, corpus_path):
        # Create the directory for storing language models
        model_directory = MODEL_DIR / self.corpus_name
        model_directory.mkdir(parents=True, exist_ok=True)

        for q in self.q_range:
            # Generate and load KenLM models for each q-gram size
            _, binary_file = run_kenlm(self.corpus_name, q, corpus_path, model_directory)
            if binary_file:
                self.model[q] = kenlm.Model(binary_file)  # Load and store the KenLM model
                logging.info(f"Model for {q}-gram loaded.")

    def generate_and_load_models(self):
        # Generate a formatted training set and build language models from it
        formatted_training_set_path = SETS_DIR / f"{self.corpus_name}_formatted_training_set.txt"
        # Format and save the training set for KenLM processing
        self.generate_formatted_corpus(self.training_set, formatted_training_set_path)
        # Generate KenLM language models from the formatted training set
        self.generate_models_from_corpus(formatted_training_set_path)

    def replace_random_letter(self, word, include_original=False) -> tuple[str, str, str]:
        # Randomly replaces a vowel or a consonant in 'word' with an underscore.

        # Gather indices of vowels and consonants for potential replacement.
        vowel_indices = [i for i, letter in enumerate(word) if letter in Letters.VOWELS.value]
        consonant_indices = [i for i, letter in enumerate(word) if letter in Letters.CONSONANTS.value]

        # Randomly decide whether to replace a vowel or consonant based on set ratios.
        random_choice = self.rng.random()
        combined_probability = self.vowel_replacement_ratio + self.consonant_replacement_ratio

        # Choose the letter type to replace based on the random choice and their availability.
        if random_choice < self.vowel_replacement_ratio and vowel_indices:
            letter_indices = vowel_indices
        elif random_choice < combined_probability and consonant_indices:
            letter_indices = consonant_indices
        else:
            letter_indices = vowel_indices if vowel_indices else consonant_indices

        # Ensure there are available letters to replace; raise an error otherwise.
        if not letter_indices:
            raise ValueError(f"Unable to replace a letter in word: '{word}'.")

        # Select a random letter from the chosen type and replace it with an underscore.
        letter_index = self.rng.choice(letter_indices)
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        # Return the modified word, the missing letter, and optionally the original word.
        return (modified_word, missing_letter, word) if include_original else (modified_word, missing_letter)

    def evaluate_model(self, prediction_method) -> tuple:
        # Initialize counters for accuracy and validity.
        accuracy_counts = defaultdict(int)
        validity_counts = defaultdict(int)
        total_test_words = len(self.test_set)
        predictions = []

        # Iterate through each word in the test set for evaluation.
        for modified_word, missing_letter, original_word in self.test_set:
            top_three_predictions = prediction_method(modified_word)
            
            # Initialize flags for each word to track if a correct or valid prediction has been found
            correct_found = False
            valid_word_found = False

            for rank, (predicted_letter, _) in enumerate(top_three_predictions, start=1):
                reconstructed_word = modified_word.replace('_', predicted_letter)

                # Update accuracy if the predicted letter is correct
                if predicted_letter == missing_letter:
                    for i in range(rank, 4):
                        accuracy_counts[i] += 1
                    correct_found = True  # Indicate that a correct prediction was found

                # Update validity if the reconstructed word exists in the corpus
                if not valid_word_found and reconstructed_word in self.all_words:
                    for i in range(rank, 4):
                        validity_counts[i] += 1
                    valid_word_found = True  # Indicate that a valid word was found

            predictions.append((modified_word, missing_letter, original_word, top_three_predictions))

        # Calculate accuracy and validity rates
        total_accuracy = {k: count / total_test_words for k, count in accuracy_counts.items()}
        total_validity = {k: count / total_test_words for k, count in validity_counts.items()}

        evaluation_metrics = {
            'accuracy': total_accuracy,
            'validity': total_validity,
            'total_words': total_test_words
        }

        return evaluation_metrics, predictions

    def save_predictions_to_csv(self, evaluation_metrics, predictions, prediction_method_name):
        # Save prediction results and related metrics to a CSV file with additional details.
        csv_file_path = CSV_DIR / f"{self.corpus_name}_predictions.csv"

        with csv_file_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Define the header with columns for prediction details and additional metrics
            writer.writerow([
                'Corpus', 'Prediction Method', 'Training Size', 'Testing Size',
                'Vowel Ratio', 'Consonant Ratio', 'Tested Word', 'Original Word', 'Correct Letter',
                'Prediction Rank', 'Predicted Letter', 'Confidence', 'Is Valid Word',
                'Missing Letter Position', 'Word Length'
            ])

            for modified_word, missing_letter, original_word, top_three_predictions in predictions:
                # Calculate the position of the missing letter and the length of the original word
                missing_letter_position = modified_word.index('_')
                word_length = len(original_word)

                for rank, (predicted_letter, confidence) in enumerate(top_three_predictions, start=1):
                    # Check if the reconstructed word (with predicted letter) is valid
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    is_valid_word = reconstructed_word in self.all_words

                    # Prepare a row with corpus details, prediction results, and additional metrics
                    row = [
                        self.corpus_name, prediction_method_name, 
                        len(self.training_set), len(self.test_set), 
                        self.vowel_replacement_ratio, self.consonant_replacement_ratio, 
                        modified_word, original_word, missing_letter,
                        rank, predicted_letter, f"{confidence:.7f}", is_valid_word,
                        missing_letter_position, word_length
                    ]

                    writer.writerow(row)

    def save_predictions_to_file(self, evaluation_metrics, predictions, file_path, prediction_method_name):
        # Save detailed predictions and metrics to a text file for easy review.
        with file_path.open('w', encoding='utf-8') as file:
            # Document the prediction method used at the top of the file
            file.write(f"Prediction Method: {prediction_method_name}\n\n")

            # Write overall accuracy and validity metrics for easy reference
            accuracy = evaluation_metrics['accuracy']
            validity = evaluation_metrics['validity']
            file.write(f"TOP1 ACCURACY: {accuracy[1]:.2%}\n")
            file.write(f"TOP2 ACCURACY: {accuracy[2]:.2%}\n")
            file.write(f"TOP3 ACCURACY: {accuracy[3]:.2%}\n")
            file.write(f"TOP1 VALIDITY: {validity[1]:.2%}\n")
            file.write(f"TOP2 VALIDITY: {validity[2]:.2%}\n")
            file.write(f"TOP3 VALIDITY: {validity[3]:.2%}\n\n")

            # Detail each prediction for individual test words
            for modified_word, missing_letter, original_word, top_three_predictions in predictions:
                file.write(f"Test Word: {modified_word}\nOriginal Word: {original_word}\nMissing Letter: {missing_letter}\n")
                for rank, (predicted_letter, confidence) in enumerate(top_three_predictions, start=1):
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    is_valid_word = reconstructed_word in self.all_words
                    file.write(f"Rank {rank}: '{predicted_letter}' (confidence {confidence:.7f}, valid: {is_valid_word})\n")
                file.write("\n")

    def save_set_to_file(self, data_set, file_name):
        # Write the contents of a data set to a file, formatting tuples for readability
        file_path = SETS_DIR / file_name
        with file_path.open('w', encoding='utf-8') as file:
            for item in data_set:
                # Format tuples with parentheses and comma separation
                formatted_line = f"({', '.join(map(str, item))})" if isinstance(item, tuple) else str(item)
                file.write(formatted_line + "\n")

def run(corpus_name, config):
    # Start processing the specified corpus
    logging.info(f"Processing {corpus_name} Corpus")
    logging.info("-" * 40)

    # Initialize the Language Model with the given corpus and configuration
    lm = LanguageModel(corpus_name, config)
    logging.info(f"{corpus_name} Language model initialized")

    # Load corpus data
    lm.load_corpus(corpus_name)
    logging.info("Corpus data loaded")

    # Prepare the training and test datasets
    lm.prepare_datasets()
    logging.info(f"Training set size: {len(lm.training_set)}")
    logging.info(f"Test set size: {len(lm.test_set)}")

    # Generate and load Q-gram models
    lm.generate_and_load_models()
    logging.info(f"{corpus_name} Q-gram models generated and loaded")

    # Select and log the prediction method being evaluated
    prediction_method = lm.predictor.context_sensitive
    logging.info(f"Evaluated with: {prediction_method.__name__}")

    # Evaluate the model and capture metrics
    evaluation_metrics, predictions = lm.evaluate_model(prediction_method)

    # Log the accuracy and validity results
    accuracy = evaluation_metrics['accuracy']
    validity = evaluation_metrics['validity']
    logging.info(f"Model evaluation completed for: {corpus_name}")
    logging.info(f"TOP1 ACCURACY: {accuracy[1]:.2%} | TOP1 VALIDITY: {validity[1]:.2%}")
    logging.info(f"TOP2 ACCURACY: {accuracy[2]:.2%} | TOP2 VALIDITY: {validity[2]:.2%}")
    logging.info(f"TOP3 ACCURACY: {accuracy[3]:.2%} | TOP3 VALIDITY: {validity[3]:.2%}")

    # Save the predictions to CSV and text files
    lm.save_predictions_to_csv(evaluation_metrics, predictions, prediction_method.__name__)
    output_file = TEXT_DIR / f"{lm.corpus_name}_predictions.txt"
    lm.save_predictions_to_file(evaluation_metrics, predictions, output_file, prediction_method.__name__)

    logging.info("-" * 40)

def main():
    config = Config()
    setup_logging()
    corpora = ["cmudict", "brown", "CLMET3.txt"]
    for corpus_name in corpora:
        run(corpus_name, config)

if __name__ == "__main__":
    main()