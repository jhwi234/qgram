import random
import logging
import re
from pathlib import Path
import enum
import subprocess
from retry import retry

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
SETS_DIR = OUTPUT_DIR / "sets"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, CORPUS_DIR, OUTPUT_DIR, SETS_DIR]:
    directory.mkdir(exist_ok=True)

# Constants and Enums
VOWELS = 'aeiouæœ'
CONSONANTS = ''.join(set('abcdefghijklmnopqrstuvwxyz') - set(VOWELS))

class LetterType(enum.Enum):
    VOWEL = 1
    CONSONANT = 2

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

# Suppress NLTK download messages
nltk.download('punkt', quiet=True)

# KenLM Wrapper Function
@retry(subprocess.CalledProcessError, tries=3, delay=2)
def run_kenlm(corpus_name, q, corpus_path, model_directory):
    # This function serves as a wrapper for the KenLM language model toolkit. KenLM efficiently
    # The function automates the process of creating models for a given corpus.

    arpa_file = model_directory / f'{corpus_name}_{q}gram.arpa'
    binary_file = model_directory / f'{corpus_name}_{q}gram.klm'

    try:
        # Runs the 'lmplz' command from KenLM to build an ARPA file.
        subprocess.run(
            ['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Converts the ARPA file to a binary format using 'build_binary'.
        subprocess.run(
            ['build_binary', '-s', str(arpa_file), str(binary_file)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return q, str(binary_file)
    except subprocess.CalledProcessError as e:
        # If an error occurs during the subprocess calls, it's logged and the error is re-raised.
        logging.error(f"Attempt to generate/load the model for {q}-gram failed: {e}")
        raise
class LanguageModel:

    CLEAN_PATTERN = re.compile(r'\b[a-zA-Z]{4,}\b')

    def __init__(self, corpus_name, q_range=(6, 6), split_config=0.5, seed=None):
        self.corpus_name = corpus_name.replace('.txt', '')
        self.q_range = range(q_range[0], q_range[1] + 1)
        self.model = {}
        self.predictor = Predictions(self.model, self.q_range) 
        self.corpus = set()
        self.test_set = set()
        self.training_set = set()
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.all_words = set()
        self.split_config = split_config

    def clean_text(self, text: str) -> set[str]:
        return set(word.lower() for word in self.CLEAN_PATTERN.findall(text))

    def load_corpus(self, corpus_name):
        # Load a corpus either from a text file or using NLTK if it's a known dataset
        if corpus_name.endswith('.txt'):
            file_path = CORPUS_DIR / corpus_name
            with file_path.open('r', encoding='utf-8') as file:
                self.corpus = self.clean_text(file.read())
        else:
            nltk.download(corpus_name, quiet=True)  # Downloads the corpus using NLTK if not already present
            self.corpus = self.clean_text(' '.join(getattr(nltk.corpus, corpus_name).words()))

    def _shuffle_and_split_corpus(self):
        # Shuffles the corpus and splits it into training and test sets based on the split configuration
        total_size = len(self.corpus)
        shuffled_corpus = list(self.corpus)
        self.rng.shuffle(shuffled_corpus) 
        training_size = int(total_size * self.split_config)  # Calculate size of the training set
        return set(shuffled_corpus[:training_size]), set(shuffled_corpus[training_size:])

    def prepare_datasets(self, vowel_replacement_ratio=0.5, consonant_replacement_ratio=0.5):
        # Prepares training and test datasets, modifying the test set to simulate missing letters
        self.training_set, unprocessed_test_set = self._shuffle_and_split_corpus()
        self.save_set_to_file(self.training_set, f"{self.corpus_name}_training_set.txt")

        # For each word in the test set, replace a letter (vowel/consonant based on provided ratios)
        self.test_set = {self.replace_random_letter(word, include_original=True, vowel_replacement_ratio=vowel_replacement_ratio, consonant_replacement_ratio=consonant_replacement_ratio) for word in unprocessed_test_set}
        self.save_set_to_file(self.test_set, f"{self.corpus_name}_formatted_test_set.txt")
        self.all_words = self.training_set.union(self.test_set)  # Combines training and test sets for word verification during testing

    def generate_formatted_corpus(self, data_set, formatted_corpus_path):
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

    def replace_random_letter(self, word, include_original=False, vowel_replacement_ratio=0.5, consonant_replacement_ratio=0.5):
        # Replaces a random letter (vowel or consonant) in a given word based on specified ratios
        # Categorize indices of vowels and consonants within the word
        categorized_indices = {
            LetterType.VOWEL: [i for i, letter in enumerate(word) if letter in VOWELS],
            LetterType.CONSONANT: [i for i, letter in enumerate(word) if letter in CONSONANTS]
        }

        # Decide randomly whether to replace a vowel or a consonant based on the given replacement ratios
        replace_vowel = random.random() < vowel_replacement_ratio and categorized_indices[LetterType.VOWEL]
        replace_consonant = random.random() < consonant_replacement_ratio and categorized_indices[LetterType.CONSONANT]

        # Select the type of letter (vowel or consonant) to replace based on random choice
        if replace_vowel and replace_consonant:
            letter_type = self.rng.choice([LetterType.VOWEL, LetterType.CONSONANT])
        elif replace_vowel:
            letter_type = LetterType.VOWEL
        elif replace_consonant:
            letter_type = LetterType.CONSONANT
        else:
            # Default fallback to select any letter type if neither vowel nor consonant is selected
            letter_type = self.rng.choice([LetterType.VOWEL, LetterType.CONSONANT])
            # Further fallback if the selected letter type does not exist in the word
            if not categorized_indices[letter_type]:
                letter_type = LetterType.VOWEL if letter_type == LetterType.CONSONANT else LetterType.CONSONANT

        # Randomly select an index from the chosen letter type to replace
        letter_indices = categorized_indices[letter_type]
        letter_index = self.rng.choice(letter_indices)

        # Replace the selected letter with a placeholder ('_') and store the original letter
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        # Returns a tuple containing the modified word, the missing letter, and optionally the original word
        if include_original:
            return (modified_word, missing_letter, word)
        else:
            return modified_word, missing_letter

    def evaluate_model(self, prediction_method):
        # Initialize counters for correct predictions at different ranks
        correct_counts = {1: 0, 2: 0, 3: 0}
        total_words = len(self.test_set)  # Total number of words in the test set
        top1_valid_predictions = 0
        top2_valid_predictions = 0
        top3_valid_predictions = 0
        predictions = []

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

            predictions.append((modified_word, missing_letter, top_three_predictions))

        # Calculate recall for TOP1, TOP2, and TOP3 predictions
        top1_recall = top1_valid_predictions / total_words if total_words > 0 else 0.0
        top2_recall = top2_valid_predictions / total_words if total_words > 0 else 0.0
        top3_recall = top3_valid_predictions / total_words if total_words > 0 else 0.0

        # Save predictions and metrics to a file
        output_file_name = f"{self.corpus_name}_predictions.txt"
        output_file = OUTPUT_DIR / output_file_name
        self.save_predictions_to_file(correct_counts, top1_recall, top2_recall, top3_recall, total_words, predictions, output_file)

        return correct_counts, top1_recall, top2_recall, top3_recall

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
            for modified_word, missing_letter, top_three_predictions in predictions:
                top_predicted_letter, top_confidence = top_three_predictions[0]
                file.write(f"Test Word: {modified_word}\nOriginal: {missing_letter}\n")
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
    def format_data_for_saving(item):
        # Converts data items into a string format suitable for saving to a file.
        if isinstance(item, tuple):
            # If the item is a tuple, format it as a string with each element separated by commas.
            return f"({item[0]}, {item[1]}, {item[2]})"
        else:
            # If the item is not a tuple (just a single word), return it as a string.
            return f"{item}"


def main():
    # Set the global random seed at the start of the main function
    seed = 42
    # Sets up logging and runs the model processing for specified corpora.
    setup_logging()
    run("cmudict", seed)
    run("brown", seed)
    run("CLMET3.txt", seed)

def run(corpus_name, seed):
    # Processes a specified corpus for language model evaluation.
    print(f"Processing {corpus_name} Corpus")
    print("-" * 40)

    lm = LanguageModel(corpus_name, seed=seed)
    lm.load_corpus(corpus_name)  # Load the corpus data
    logging.info(f"{corpus_name} Language model initialized")

    # Retrieve default parameters for vowel and consonant replacement ratios.
    vowel_ratio, consonant_ratio = LanguageModel.prepare_datasets.__defaults__[-2:]
    logging.info(f"Vowel Replacement Ratio: {vowel_ratio}")
    logging.info(f"Consonant Replacement Ratio: {consonant_ratio}")

    lm.prepare_datasets()  # Prepare the datasets for training and testing
    logging.info(f"Training set size: {len(lm.training_set)}")
    logging.info(f"Test set size: {len(lm.test_set)}")

    lm.generate_and_load_models()  # Generate and load language models
    logging.info(f"{corpus_name} Q-gram models generated and loaded")

    prediction_method = lm.predictor.context_sensitive_prediction
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
    print("-" * 40)

if __name__ == "__main__":
    main()