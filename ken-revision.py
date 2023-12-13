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

        self.split_config = split_config

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

    def evaluate_model(self, prediction_method) -> tuple[dict[int, int], float, float, float]:
        # Initialize counters for correct predictions at different ranks
        correct_counts = {1: 0, 2: 0, 3: 0}
        total_test_words = len(self.test_set)
        top1_valid_predictions = 0
        top2_valid_predictions = 0
        top3_valid_predictions = 0
        predictions = []
        prediction_method_name = prediction_method.__name__

        for test_data in self.test_set:
            # Process each word in the test set
            modified_word, missing_letter, original_word = test_data
            top_three_predictions = prediction_method(modified_word)  # Get the top three predictions

            correct_found = False
            for rank, (predicted_letter, _) in enumerate(top_three_predictions, start=1):
                reconstructed_word = modified_word.replace('_', predicted_letter)
                if predicted_letter == missing_letter:
                    # If the correct letter is predicted, update the counters
                    correct_found = True
                    for i in range(rank, 4):
                        correct_counts[i] += 1
                    # Update valid predictions counters based on rank
                    if rank == 1: top1_valid_predictions += 1
                    if rank <= 2: top2_valid_predictions += 1
                    top3_valid_predictions += 1
                    break

                # Check if the predicted word exists in the corpus for TOP1 and TOP2 predictions
                if rank == 1 and reconstructed_word in self.all_words: top1_valid_predictions += 1
                if rank == 2 and reconstructed_word in self.all_words: top2_valid_predictions += 1

            # Check for any valid word in TOP3 predictions if correct letter wasn't found
            if not correct_found:
                if any(modified_word.replace('_', pred_letter) in self.all_words for pred_letter, _ in top_three_predictions):
                    top3_valid_predictions += 1

            predictions.append((modified_word, missing_letter, original_word, top_three_predictions))

        # Calculate recall for TOP1, TOP2, and TOP3 predictions
        top1_recall = top1_valid_predictions / total_test_words if total_test_words > 0 else 0.0
        top2_recall = top2_valid_predictions / total_test_words if total_test_words > 0 else 0.0
        top3_recall = top3_valid_predictions / total_test_words if total_test_words > 0 else 0.0

        # Save predictions and metrics to a file
        output_file_name = f"{self.corpus_name}_predictions.txt"
        output_file = TEXT_DIR / output_file_name
        self.save_predictions_to_file(correct_counts, top1_recall, top2_recall, top3_recall, total_test_words, predictions, output_file)
        self.save_predictions_to_csv(predictions, prediction_method_name)
        return correct_counts, top1_recall, top2_recall, top3_recall

    def save_predictions_to_csv(self, predictions, prediction_method_name):
        csv_file_path = CSV_DIR / f"{self.corpus_name}_predictions.csv"
        
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

    def save_predictions_to_file(self, correct_counts, top1_recall, top2_recall, top3_recall, total_words, predictions, file_path):
        # Writes prediction results to a specified file
        with file_path.open('w', encoding='utf-8') as file:
            # Write accuracies and recall metrics
            file.write(f"TOP1 PRECISION: {correct_counts[1] / total_words:.2%}\n")
            file.write(f"TOP2 PRECISION: {correct_counts[2] / total_words:.2%}\n")
            file.write(f"TOP3 PRECISION: {correct_counts[3] / total_words:.2%}\n")
            file.write(f"TOP1 RECALL: {top1_recall:.2%}\n")
            file.write(f"TOP2 RECALL: {top2_recall:.2%}\n")
            file.write(f"TOP3 RECALL: {top3_recall:.2%}\n\n")

            # Write detailed predictions for each test word
            for modified_word, missing_letter, original_word, top_three_predictions in predictions:
                top_predicted_letter, top_confidence = top_three_predictions[0]
                file.write(f"Test Word: {modified_word}\nOriginal Word: {original_word}\nMissing Letter: {missing_letter}\n")
                file.write(f"Predicted: {top_predicted_letter}\n")
                file.write("Top Three Predictions:\n")
                for rank, (letter, confidence) in enumerate(top_three_predictions, start=1):
                    file.write(f"Rank {rank}: '{letter}' confidence {confidence:.7f}\n")
                file.write("\n")

    def save_set_to_file(self, data_set, file_name):
        # Saves a given data set to a file
        file_path = SETS_DIR / file_name
        with file_path.open('w', encoding='utf-8') as file:
            for item in data_set:
                # Format each item in the data set and write it to the file
                formatted_line = self.format_data_for_saving(item)
                file.write(formatted_line + "\n")

    @staticmethod
    def format_data_for_saving(item) -> str:
        # Converts data items into a string format suitable for saving to a file.
        if isinstance(item, tuple):
            # If the item is a tuple, format it as a string with each element separated by commas.
            return f"({item[0]}, {item[1]}, {item[2]})"
        else:
            # If the item is not a tuple (just a single word), return it as a string.
            return f"{item}"

def run(corpus_name, config):
    logging.info(f"Processing {corpus_name} Corpus")
    logging.info("-" * 40)

    lm = LanguageModel(corpus_name, config)
    lm.load_corpus(corpus_name)  # Load the corpus data
    logging.info(f"{corpus_name} Language model initialized")

    lm.prepare_datasets()  # Prepare the datasets for training and testing
    logging.info(f"Training set size: {len(lm.training_set)}")
    logging.info(f"Test set size: {len(lm.test_set)}")

    lm.generate_and_load_models()  # Generate and load language models
    logging.info(f"{corpus_name} Q-gram models generated and loaded")

    prediction_method = lm.predictor.context_sensitive
    logging.info(f"Evaluated with: {prediction_method.__name__}")

    # Evaluate the model and log the results
    correct_counts, top1_recall, top2_recall, top3_recall = lm.evaluate_model(prediction_method)
    logging.info(f"Model evaluation completed for: {corpus_name}")
    logging.info(f"TOP1 PRECISION: {correct_counts[1] / len(lm.test_set):.2%}")
    logging.info(f"TOP2 PRECISION: {correct_counts[2] / len(lm.test_set):.2%}")
    logging.info(f"TOP3 PRECISION: {correct_counts[3] / len(lm.test_set):.2%}")
    logging.info(f"TOP1 RECALL: {top1_recall:.2%}")
    logging.info(f"TOP2 RECALL: {top2_recall:.2%}")
    logging.info(f"TOP3 RECALL: {top3_recall:.2%}")
    logging.info("-" * 40)

def main():
    config = Config()
    setup_logging()
    corpora = ["cmudict", "brown", "CLMET3.txt"]
    for corpus_name in corpora:
        run(corpus_name, config)

if __name__ == "__main__":
    main()