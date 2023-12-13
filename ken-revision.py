import random
import logging
import re
import csv
from pathlib import Path
import subprocess

import nltk
import kenlm
from prediction_methods import Predictions

# Directory Setup
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
CORPUS_DIR = DATA_DIR / "corpora"
OUTPUT_DIR = DATA_DIR / "outputs"
TEXT_DIR = OUTPUT_DIR / "texts"
CSV_DIR = OUTPUT_DIR / "csv"
SETS_DIR = OUTPUT_DIR / "sets"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, CORPUS_DIR, OUTPUT_DIR, SETS_DIR, TEXT_DIR, CSV_DIR]:
    directory.mkdir(exist_ok=True)

# Logging Configuration
def setup_logging():
    logfile = LOG_DIR / 'logfile.log'
    file_handler = logging.FileHandler(logfile, mode='a')
    file_format = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# Constants for vowels and consonants
VOWELS = 'aeiouæœ'
CONSONANTS = 'bcdfghjklmnpqrstvwxyz'

# Ensure the unique characters in both sets (especially for extended characters like æ, œ)
ALL_LETTERS = set('abcdefghijklmnopqrstuvwxyzæœ')
CONSONANTS = ''.join(ALL_LETTERS - set(VOWELS))

# KenLM Wrapper Function
def run_kenlm(corpus_name, q, corpus_path, model_directory) -> tuple[int, str]:

    # Automates the creation of ARPA and binary language models for a given corpus.
    arpa_file = model_directory / f'{corpus_name}_{q}gram.arpa'
    binary_file = model_directory / f'{corpus_name}_{q}gram.klm'

    # Run 'lmplz' to build an ARPA file and 'build_binary' to convert it to binary format
    subprocess.run(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['build_binary', '-s', str(arpa_file), str(binary_file)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return q, str(binary_file)
class Config:
    """
    Configuration class for setting up language model testing parameters.

    Attributes:
    - seed (int): Seed for random number generator to ensure reproducibility.
    - q_range (tuple): Range of q-grams to use for the language model.
    - split_config (float): Proportion of corpus to use for training; remainder for testing.
    - vowel_replacement_ratio (float): Probability of replacing vowels in test data.
    - consonant_replacement_ratio (float): Probability of replacing consonants in test data.
    - min_word_length (int): Minimum length of word strings to be included in the word set.
    """

    def __init__(self, seed: int = 42, q_range: tuple = (6, 6), split_config: float = 0.5,
                 vowel_replacement_ratio: float = 0.5, consonant_replacement_ratio: float = 0.5, 
                 min_word_length: int = 4):
        
        self.seed = seed
        self.q_range = q_range
        self.split_config = 0.9
        self.vowel_replacement_ratio = vowel_replacement_ratio
        self.consonant_replacement_ratio = consonant_replacement_ratio
        self.min_word_length = min_word_length
class LanguageModel:

    CLEAN_PATTERN = re.compile(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b')

    def __init__(self, corpus_name, config):
        self.corpus_name = corpus_name.replace('.txt', '')
        self.config = config
        self.q_range = range(config.q_range[0], config.q_range[1] + 1)
        self.model = {}
        self.predictor = Predictions(self.model, self.q_range) 
        self.corpus = set()
        self.test_set = set()
        self.training_set = set()
        self.rng = random.Random(config.seed)
        self.all_words = set()
        self.split_config = config.split_config
        self.vowel_replacement_ratio = config.vowel_replacement_ratio
        self.consonant_replacement_ratio = config.consonant_replacement_ratio
        self.min_word_length = config.min_word_length

        logging.info(f"Language Model for {self.corpus_name} initialized with:")
        logging.info(f"Seed: {config.seed}")
        logging.info(f"Q-gram Range: {config.q_range}")
        logging.info(f"Train-Test Split Configuration: {self.split_config}")
        logging.info(f"Vowel Replacement Ratio: {self.vowel_replacement_ratio}")
        logging.info(f"Consonant Replacement Ratio: {self.consonant_replacement_ratio}")
        logging.info(f"Minimum Word Length: {self.min_word_length}")

    def clean_text(self, text: str) -> set[str]:
    # Directly filter and process words in a single comprehension
        return {part.lower() 
                for word in self.CLEAN_PATTERN.findall(text)
                # Split hyphenated words and add all parts to the list 
                for part in word.split('-')
                # Filter out words that do not meet the minimum length requirement
                if len(part) >= self.min_word_length}

    def load_corpus(self, corpus_name):
        # Loads the corpus data from a file or NLTK
        if corpus_name.endswith('.txt'):
            file_path = CORPUS_DIR / corpus_name # CLMET3.txt is the only corpus that is not in the NLTK library
            with file_path.open('r', encoding='utf-8') as file:
                # Read and process the contents of a text file
                self.corpus = self.clean_text(file.read())
        else:
            # Download and process a standard NLTK corpus
            nltk.download(corpus_name, quiet=True)
            self.corpus = self.clean_text(' '.join(getattr(nltk.corpus, corpus_name).words()))

    def _shuffle_and_split_corpus(self) -> tuple[set[str], set[str]]:
        # Shuffles the corpus and splits it into training and test sets based on the split configuration
        total_size = len(self.corpus)
        shuffled_corpus = list(self.corpus)
        self.rng.shuffle(shuffled_corpus) 
        training_size = int(total_size * self.split_config)  # Calculate size of the training set
        return set(shuffled_corpus[:training_size]), set(shuffled_corpus[training_size:])

    def prepare_datasets(self):
        # Prepares training and test datasets, modifying the test set to simulate missing letters
        self.training_set, unprocessed_test_set = self._shuffle_and_split_corpus()
        self.save_set_to_file(self.training_set, f"{self.corpus_name}_training_set.txt")

        # For each word in the test set, replace a letter based on the class attributes for vowel and consonant replacement ratios
        self.test_set = {self.replace_random_letter(word, include_original=True) for word in unprocessed_test_set}
        self.save_set_to_file(self.test_set, f"{self.corpus_name}_formatted_test_set.txt")

        # Extract original words from the test_set tuples and combine with training_set for word verification during testing
        self.all_words = self.training_set.union({original_word for _, _, original_word in self.test_set})
        # Save the combined set to a file
        self.save_set_to_file(self.all_words, f"{self.corpus_name}_all_words.txt")

    def generate_formatted_corpus(self, data_set, formatted_corpus_path) -> Path:
        # Formats and saves the dataset to a file, with each word separated by a space and each line representing a single word
        formatted_text = [" ".join(word) for word in data_set]
        formatted_corpus = '\n'.join(formatted_text)

        with formatted_corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_corpus)

        return formatted_corpus_path

    def generate_models_from_corpus(self, corpus_path):
        # Builds and saves KenLM language models for each q-gram range specified
        model_directory = MODEL_DIR / self.corpus_name
        model_directory.mkdir(parents=True, exist_ok=True)  # Ensures the model directory exists

        for q in self.q_range:
            _, binary_file = run_kenlm(self.corpus_name, q, corpus_path, model_directory)  # Generates n-gram models
            if binary_file:
                self.model[q] = kenlm.Model(binary_file)  # Loads the generated KenLM model
                logging.info(f"Model for {q}-gram loaded.")

    def generate_and_load_models(self):
        # Orchestrates the generation of formatted corpus and subsequent language model creation
        formatted_training_set_path = SETS_DIR / f"{self.corpus_name}_formatted_training_set.txt"
        self.generate_formatted_corpus(self.training_set, formatted_training_set_path)  # Formats and saves the training set

        self.generate_models_from_corpus(formatted_training_set_path)  # Generates language models from the formatted training set

    def prepare_and_save_test_set(self, data_set, file_name):
        # Prepares the test set by modifying each word and saving it to a file
        formatted_test_set = []
        for word in data_set:
            modified_word, missing_letter = self.replace_random_letter(word)
            # Creates a tuple with the modified word (with a letter replaced by '_'), the missing letter, and the original word
            formatted_test_set.append((modified_word, missing_letter, word))

        self.save_set_to_file(formatted_test_set, file_name)  # Saves the modified test set to a specified file

    def replace_random_letter(self, word, include_original=False) -> tuple[str, str, str]:
        # Randomly replaces a vowel or a consonant in a word with an underscore (_).

        # Collect indices of vowels and consonants in the word.
        vowel_indices = [i for i, letter in enumerate(word) if letter in VOWELS]
        consonant_indices = [i for i, letter in enumerate(word) if letter in CONSONANTS]

        # Generate a random number to decide whether to replace a vowel or a consonant.
        # The probability of replacing a vowel or consonant is determined by the predefined ratios.
        random_choice = self.rng.random()
        combined_probability = self.vowel_replacement_ratio + self.consonant_replacement_ratio

        # Determine the indices for replacement based on the random choice and availability of letter types.
        if random_choice < self.vowel_replacement_ratio and vowel_indices:
            letter_indices = vowel_indices
        elif random_choice < combined_probability and consonant_indices:
            letter_indices = consonant_indices
        else:
            # Fallback: Use whichever type of letters is available.
            letter_indices = vowel_indices if vowel_indices else consonant_indices

        # Handle cases where no suitable letter is available for replacement.
        if not letter_indices:
            raise ValueError(f"Unable to replace a letter in word: '{word}'.")

        # Randomly select one of the available letters to replace with '_'.
        letter_index = self.rng.choice(letter_indices)
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        # Return the modified word with the missing letter, and optionally the original word.
        return (modified_word, missing_letter, word) if include_original else (modified_word, missing_letter)

    def evaluate_model(self, prediction_method) -> tuple:
        # Initialize metrics for accuracy and validity
        accuracy_counts = {1: 0, 2: 0, 3: 0}
        validity_counts = {1: 0, 2: 0, 3: 0}
        total_test_words = len(self.test_set)
        predictions = []

        for test_data in self.test_set:
            modified_word, missing_letter, original_word = test_data
            top_three_predictions = prediction_method(modified_word)
            correct_found = False
            valid_word_found = False

            for rank, (predicted_letter, _) in enumerate(top_three_predictions, start=1):
                reconstructed_word = modified_word.replace('_', predicted_letter)

                # Update accuracy if the predicted letter is correct
                if predicted_letter == missing_letter:
                    for i in range(rank, 4):
                        accuracy_counts[i] += 1

                # Update validity if the reconstructed word exists in the corpus and a valid word hasn't been found yet
                if not valid_word_found and reconstructed_word in self.all_words:
                    for i in range(rank, 4):
                        validity_counts[i] += 1
                    valid_word_found = True

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
        csv_file_path = CSV_DIR / f"{self.corpus_name}_predictions.csv"
        total_words = evaluation_metrics['total_words']
        
        with csv_file_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
             # Writing the header with separate columns for letter and confidence
            writer.writerow([
                'Corpus', 'Prediction Method', 'Training Size', 'Testing Size',
                'Vowel Ratio', 'Consonant Ratio', 'Tested Word', 'Original Word', 'Correct Letter',
                'Top1 Letter', 'Top1 Confidence', 'Top1 Validity',
                'Top2 Letter', 'Top2 Confidence', 'Top2 Validity',
                'Top3 Letter', 'Top3 Confidence', 'Top3 Validity'
            ])

            # Writing the data rows with separate columns for letter and confidence
            for modified_word, missing_letter, original_word, top_three_predictions in predictions:
                row = [
                    self.corpus_name, prediction_method_name, 
                    len(self.training_set), len(self.test_set), 
                    self.vowel_replacement_ratio, self.consonant_replacement_ratio, 
                    modified_word, original_word, missing_letter
                ]

                # Adding predictions with separated letter and confidence, formatted consistently
                for predicted_letter, confidence in top_three_predictions:
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    is_valid_word = reconstructed_word in self.all_words
                    row.extend([predicted_letter, f"{confidence:.7f}", is_valid_word])

                writer.writerow(row)

    def save_predictions_to_file(self, evaluation_metrics, predictions, file_path, prediction_method_name):
        with file_path.open('w', encoding='utf-8') as file:
            # Write the prediction method name at the top
            file.write(f"Prediction Method: {prediction_method_name}\n\n")

            # Unpack accuracy and validity metrics
            accuracy = evaluation_metrics['accuracy']
            validity = evaluation_metrics['validity']
            total_words = evaluation_metrics['total_words']

            # Write overall accuracy and validity metrics
            file.write(f"TOP1 ACCURACY: {accuracy[1]:.2%}\n")
            file.write(f"TOP2 ACCURACY: {accuracy[2]:.2%}\n")
            file.write(f"TOP3 ACCURACY: {accuracy[3]:.2%}\n")
            file.write(f"TOP1 VALIDITY: {validity[1]:.2%}\n")
            file.write(f"TOP2 VALIDITY: {validity[2]:.2%}\n")
            file.write(f"TOP3 VALIDITY: {validity[3]:.2%}\n\n")

            # Write detailed predictions for each test word
            for modified_word, missing_letter, original_word, top_three_predictions in predictions:
                file.write(f"Test Word: {modified_word}\nOriginal Word: {original_word}\nMissing Letter: {missing_letter}\n")
                for rank, (predicted_letter, confidence) in enumerate(top_three_predictions, start=1):
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    is_valid_word = reconstructed_word in self.all_words
                    file.write(f"Rank {rank}: '{predicted_letter}' (confidence {confidence:.7f}, valid: {is_valid_word})\n")
                file.write("\n")

    def save_set_to_file(self, data_set, file_name):
        # Saves a given data set to a file
        file_path = SETS_DIR / file_name
        with file_path.open('w', encoding='utf-8') as file:
            for item in data_set:
                # Check if the item is a tuple and format it accordingly
                if isinstance(item, tuple):
                    formatted_line = f"({', '.join(map(str, item))})"
                else:
                    formatted_line = str(item)
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