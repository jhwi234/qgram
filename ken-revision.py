import random
import logging
import regex as reg
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
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = DATA_DIR / 'models'
LOG_DIR = DATA_DIR / 'logs'
CORPUS_DIR = DATA_DIR / 'corpora'
OUTPUT_DIR = DATA_DIR / 'outputs'
TEXT_DIR = OUTPUT_DIR / 'texts'
CSV_DIR = OUTPUT_DIR / 'csv'
SETS_DIR = OUTPUT_DIR / 'sets'

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
def build_kenlm_model(corpus_name, q, corpus_path, model_directory) -> tuple[int, str]:
    arpa_file = model_directory / f'{corpus_name}_{q}gram.arpa'
    binary_file = model_directory / f'{corpus_name}_{q}gram.klm'

    # Build ARPA file and convert it to binary format for efficient usage
    subprocess.run(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['build_binary', '-s', str(arpa_file), str(binary_file)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Return q-gram size and path to the binary model file
    return q, str(binary_file)

# Configuration class for language model testing parameters. Change the testing inputs here.
class Config:
    def __init__(self, seed: int = 42, q_range: tuple = (6, 6), split_config: float = 0.5,
                 vowel_replacement_ratio: float = 0.5, consonant_replacement_ratio: float = 0.5, 
                 min_word_length: int = 4, prediction_method_name: str = 'context_sensitive'):
        self.seed = seed
        self.q_range = q_range
        self.split_config = split_config
        self.vowel_replacement_ratio = vowel_replacement_ratio
        self.consonant_replacement_ratio = consonant_replacement_ratio
        self.min_word_length = min_word_length
        self.prediction_method_name = prediction_method_name

# Language model class for processing and predicting text
class LanguageModel:
    # Regex pattern for extracting words, including hyphenated words, in various scripts.
    # \b indicates word boundaries.
    # \p{L}+ matches one or more Unicode letters, covering a wide range of characters beyond ASCII.
    # (?:-\p{L}+)* allows for optional hyphenated parts, matching additional Unicode letters after a hyphen.
    CLEAN_PATTERN = reg.compile(r'\b\p{L}+(?:-\p{L}+)*\b')

    def __init__(self, corpus_name, config):
        # Remove '.txt' extension from corpus name if present
        self.corpus_name = corpus_name.replace('.txt', '')
        # Store configuration parameters
        self.config = config
        # Define range of q-grams to be used in the model
        self.q_range = range(config.q_range[0], config.q_range[1] + 1)
        # Initialize dictionary to store models for each q-gram
        self.model = {}
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

        # Load corpus data and extract unique characters
        self.load_corpus(corpus_name)
        unique_characters = self.extract_unique_characters()
        # Store the count of unique characters
        self.unique_character_count = len(unique_characters)
        # Initialize dictionaries for tracking occurrences and correct predictions of each character
        self.character_occurrences = {char: 0 for char in unique_characters}
        self.correct_predictions = {char: 0 for char in unique_characters}

        # Initialize prediction class with models, q-gram range, and unique characters
        self.predictor = Predictions(self.model, self.q_range, unique_characters)

        # Log initialization information
        logging.info(f'Language Model for {self.corpus_name} initialized with:')
        logging.info(f'Seed: {config.seed}')
        logging.info(f'Q-gram Range: {config.q_range}')
        logging.info(f'Train-Test Split Configuration: {self.split_config}')
        logging.info(f'Vowel Replacement Ratio: {self.vowel_replacement_ratio}')
        logging.info(f'Consonant Replacement Ratio: {self.consonant_replacement_ratio}')
        logging.info(f'Unique Character Count: {self.unique_character_count}')
        logging.info(f'Minimum Word Length: {self.min_word_length}')

        # Retrieve the prediction method based on the name
        prediction_methods = {
            'context_sensitive': self.predictor.context_sensitive,
            'context_no_boundary': self.predictor.context_no_boundary,
            'base_prediction': self.predictor.base_prediction
        }
        self.prediction_method = prediction_methods.get(config.prediction_method_name, self.predictor.context_sensitive)

    def extract_unique_characters(self) -> set:
        # Extracts unique characters from the corpus
        unique_chars = set()
        for word in self.corpus:
            unique_chars.update(word)
        return unique_chars

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
        self.save_set_to_file(self.training_set, f'{self.corpus_name}_training_set.txt')

        # Process the test set by replacing a letter in each word with an underscore
        formatted_test_set = []
        for word in unprocessed_test_set:
            modified_word, missing_letter = self.replace_random_letter(word)
            formatted_test_set.append((modified_word, missing_letter, word))

        self.test_set = set(formatted_test_set)
        self.save_set_to_file(self.test_set, f'{self.corpus_name}_formatted_test_set.txt')
        # Combine training and test sets for a comprehensive word list
        self.all_words = self.training_set.union({original_word for _, _, original_word in self.test_set})
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
        model_directory = MODEL_DIR / self.corpus_name
        model_directory.mkdir(parents=True, exist_ok=True)

        for q in self.q_range:
            # Generate and load KenLM models for each q-gram size
            _, binary_file = build_kenlm_model(self.corpus_name, q, corpus_path, model_directory)
            if binary_file:
                self.model[q] = kenlm.Model(binary_file)  # Load and store the KenLM model
                logging.info(f'Model for {q}-gram loaded.')

    def generate_and_load_models(self):
        # Generate a formatted training set and build language models from it
        formatted_training_set_path = SETS_DIR / f'{self.corpus_name}_formatted_training_set.txt'
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

    def compute_accuracy(self, predictions):
        # Initialize a dictionary to track accuracy counts for TOP1, TOP2, and TOP3
        accuracy_counts = defaultdict(int)
        total_test_words = len(self.test_set)

        for _, missing_letter, _, all_predictions, _ in predictions:
            # Find the highest rank at which the correct prediction is made
            correct_rank = next((rank for rank, (predicted_letter, _) in enumerate(all_predictions, start=1) if predicted_letter == missing_letter), None)
            
            # Update accuracy counts for all ranks up to the correct rank
            if correct_rank:
                for rank in range(correct_rank, 4):
                    accuracy_counts[rank] += 1

        # Calculate total accuracy for each rank
        total_accuracy = {k: accuracy_counts[k] / total_test_words for k in range(1, 4)}
        return total_accuracy

    def compute_validity(self, predictions):
        # Dictionary to store validity counts for TOP1, TOP2, and TOP3
        validity_counts = defaultdict(int)
        total_test_words = len(self.test_set)

        for modified_word, _, _, all_predictions, _ in predictions:
            # Flag to indicate if a valid word has been found
            valid_word_found = False

            for rank, (predicted_letter, _) in enumerate(all_predictions, start=1):
                if not valid_word_found:
                    # Reconstruct word with the predicted letter
                    reconstructed_word = modified_word.replace('_', predicted_letter)

                    # Check if reconstructed word is valid and update counts
                    if reconstructed_word in self.all_words:
                        for i in range(rank, 4):
                            validity_counts[i] += 1
                        valid_word_found = True

        # Calculate total validity for each rank
        total_validity = {k: count / total_test_words for k, count in validity_counts.items()}
        return total_validity

    def compute_recall(self):
        # Calculate recall for each character in the corpus
        recall_metrics = {
            char: (self.correct_predictions[char] / self.character_occurrences[char] if self.character_occurrences[char] > 0 else 0)
            for char in self.character_occurrences
        }
        return recall_metrics

    def evaluate_predictions(self, prediction_method) -> tuple[dict, list]:
        predictions = []

        for modified_word, missing_letter, original_word in self.test_set:
            self.character_occurrences[missing_letter] += 1

            all_predictions = prediction_method(modified_word)

            # Determine the rank at which the correct letter appears
            correct_letter_rank = next((rank for rank, (predicted_letter, _) in enumerate(all_predictions, start=1) if predicted_letter == missing_letter), None)
            
            # Update correct predictions count if the correct letter is within the top 3 predictions
            if correct_letter_rank and correct_letter_rank <= 3:
                self.correct_predictions[missing_letter] += 1

            predictions.append((modified_word, missing_letter, original_word, all_predictions[:3], correct_letter_rank))

        total_accuracy = self.compute_accuracy(predictions)
        total_validity = self.compute_validity(predictions)
        total_recall = self.compute_recall()

        evaluation_metrics = {
            'accuracy': total_accuracy,
            'validity': total_validity,
            'recall': total_recall,
            'total_words': len(self.test_set)
        }
        return evaluation_metrics, predictions

    def save_recall_stats(self):
        # Calculate recall metrics
        recall_metrics = self.compute_recall()

        # Sort recall metrics by recall value in descending order
        sorted_recall_metrics = sorted(recall_metrics.items(), key=lambda item: item[1], reverse=True)

        # Path for recall metrics file
        recall_file_path = TEXT_DIR / f'{self.corpus_name}_recall_metrics.txt'

        # Write sorted recall metrics to file
        with recall_file_path.open('w', encoding='utf-8') as file:
            file.write('Character, Total Occurrences, Correct Predictions, Recall\n')
            for char, recall in sorted_recall_metrics:
                total_occurrences = self.character_occurrences[char]
                correct_predictions = self.correct_predictions[char]
                file.write(f'{char}, {total_occurrences}, {correct_predictions}, {recall:.4f}\n')

        logging.info(f'Recall metrics saved to {recall_file_path}')

    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
        # Adjust file name to include test-train split and q-gram range
        csv_file_path = CSV_DIR / f'{self.corpus_name}_{prediction_method_name}_split{self.config.split_config}_qrange{self.config.q_range[0]}-{self.config.q_range[1]}_prediction.csv'

        with csv_file_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Reordered columns
            writer.writerow(['Tested Word', 'Original Word', 'Missing Letter Position', 'Word Length', 
                            'Correct Letter', 'Predicted Letter', 'Prediction Rank', 'Confidence', 
                            'Correct Letter Rank', 'Is Valid', 'Is Accurate'])
            
            for modified_word, missing_letter, original_word, top_three_predictions, correct_letter_rank in predictions:
                missing_letter_position = modified_word.index('_') + 1
                word_length = len(original_word)
                for rank, (predicted_letter, confidence) in enumerate(top_three_predictions, start=1):
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    is_valid = 1 if reconstructed_word in self.all_words else 0
                    is_accurate = 1 if predicted_letter == missing_letter else 0

                    # Reordered row
                    writer.writerow([modified_word, original_word, missing_letter_position, word_length,
                                    missing_letter, predicted_letter, rank, confidence,
                                    correct_letter_rank, is_valid, is_accurate])


    def save_summary_stats_txt(self, evaluation_metrics, predictions, prediction_method_name):
        # File path for saving prediction summary
        output_file_path = TEXT_DIR / f'{self.corpus_name}_predictions.txt'

        # Write prediction summary and metrics to text file
        with output_file_path.open('w', encoding='utf-8') as file:
            # Prediction method and unique character count
            file.write(f'Prediction Method: {prediction_method_name}\n')
            file.write(f'Unique Character Count: {self.unique_character_count}\n\n')

            # Accuracy and validity metrics
            accuracy = evaluation_metrics['accuracy']
            validity = evaluation_metrics['validity']
            file.write(f'TOP1 ACCURACY: {accuracy[1]:.2%}\n')
            file.write(f'TOP2 ACCURACY: {accuracy[2]:.2%}\n')
            file.write(f'TOP3 ACCURACY: {accuracy[3]:.2%}\n')
            file.write(f'TOP1 VALIDITY: {validity[1]:.2%}\n')
            file.write(f'TOP2 VALIDITY: {validity[2]:.2%}\n')
            file.write(f'TOP3 VALIDITY: {validity[3]:.2%}\n\n')

            # Configuration details
            file.write(f'Training Size: {len(self.training_set)}, Testing Size: {len(self.test_set)}\n')
            file.write(f'Vowel Ratio: {self.vowel_replacement_ratio}, Consonant Ratio: {self.consonant_replacement_ratio}\n\n')

            # Detailed prediction results
            for modified_word, missing_letter, original_word, top_three_predictions, correct_letter_rank in predictions:
                file.write(f'Tested Word: {modified_word}, Original Word: {original_word}, Correct Letter: {missing_letter}\n')
                file.write(f'Rank of Correct Letter: {correct_letter_rank}\n')

                for rank, (predicted_letter, confidence) in enumerate(top_three_predictions, start=1):
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    is_valid_word = reconstructed_word in self.all_words

                    file.write(f"Rank {rank}: '{predicted_letter}' (Confidence: {confidence:.8f}), Valid: {is_valid_word}\n")
                
                file.write('\n')

    def save_set_to_file(self, data_set, file_name):
        # Write the contents of a data set to a file, formatting tuples for readability
        file_path = SETS_DIR / file_name
        with file_path.open('w', encoding='utf-8') as file:
            for item in data_set:
                # Format tuples with parentheses and comma separation
                formatted_line = f"({', '.join(map(str, item))})" if isinstance(item, tuple) else str(item)
                file.write(formatted_line + '\n')

def run(corpus_name, config):
    # Start processing the specified corpus
    logging.info(f'Processing {corpus_name} Corpus')
    logging.info('-' * 40)

    # Initialize the Language Model with the given corpus and configuration
    lm = LanguageModel(corpus_name, config)
    logging.info(f'{corpus_name} Language model initialized')

    # Load corpus data
    lm.load_corpus(corpus_name)
    logging.info('Corpus data loaded')

    # Prepare the training and test datasets
    lm.prepare_datasets()
    logging.info(f'Training set size: {len(lm.training_set)}')
    logging.info(f'Test set size: {len(lm.test_set)}')

    # Generate and load Q-gram models
    lm.generate_and_load_models()
    logging.info(f'{corpus_name} Q-gram models generated and loaded')

    # Retrieve the prediction method from the Predictions object
    prediction_method = getattr(lm.predictor, config.prediction_method_name)
    evaluation_metrics, predictions = lm.evaluate_predictions(prediction_method)
    logging.info(f'Evaluated with: {prediction_method.__name__}')

    # Log the accuracy and validity results
    accuracy = evaluation_metrics['accuracy']
    validity = evaluation_metrics['validity']
    logging.info(f'Model evaluation completed for: {corpus_name}')
    logging.info(f'TOP1 ACCURACY: {accuracy[1]:.2%} | TOP1 VALIDITY: {validity[1]:.2%}')
    logging.info(f'TOP2 ACCURACY: {accuracy[2]:.2%} | TOP2 VALIDITY: {validity[2]:.2%}')
    logging.info(f'TOP3 ACCURACY: {accuracy[3]:.2%} | TOP3 VALIDITY: {validity[3]:.2%}')

    # Save the predictions to CSV and text files
    lm.export_prediction_details_to_csv(predictions, prediction_method.__name__)
    lm.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)
    lm.save_recall_stats()

    logging.info('-' * 40)

def main():
    config = Config()
    setup_logging()
    corpora = ['cmudict', 'brown', 'CLMET3.txt']
    for corpus_name in corpora:
        run(corpus_name, config)

if __name__ == '__main__':
    main()