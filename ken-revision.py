import random
import logging
import regex as reg
import csv
from pathlib import Path
import subprocess
from enum import Enum

import nltk
import kenlm
from prediction_methods import Predictions

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
        self.min_word_length = 4
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
        self.training_set = set()
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
        training_size = int(total_size * self.config.split_config)  # Calculate the size of the training set
        # Split the shuffled corpus into training and test sets and return
        return set(shuffled_corpus[:training_size]), set(shuffled_corpus[training_size:])

    def prepare_datasets(self) -> tuple[set[str], set[str]]:
        # Prepare training and test datasets from the corpus
        self.training_set, unprocessed_test_set = self._shuffle_and_split_corpus()

        # Save the formatted training set for KenLM
        formatted_training_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_training_set.txt'
        self.generate_formatted_corpus(self.training_set, formatted_training_set_path)

        # Process the test set by replacing a letter in each word with an underscore
        formatted_test_set = []
        for word in unprocessed_test_set:
            modified_word, missing_letter, _ = self.replace_random_letter(word)
            formatted_test_set.append((modified_word, missing_letter, word))

        self.test_set = set(formatted_test_set)
        self.all_words = self.training_set.union({original_word for _, _, original_word in self.test_set})

        # Save additional sets in debug mode, including the regular training set
        if self.debug:
            self.save_set_to_file(self.training_set, f'{self.corpus_name}_training_set.txt')
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
                formatted_training_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_training_set.txt'
                self.generate_formatted_corpus(self.training_set, formatted_training_set_path)
                self.generate_models_from_corpus(formatted_training_set_path)

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

class EvaluateModel:
    def __init__(self, corpus_manager):
        self.corpus_manager = corpus_manager
        self.corpus_name = corpus_manager.corpus_name 

        self.config = corpus_manager.config
        self.model = corpus_manager.model  # Use the loaded models

        # Access datasets directly from the provided corpus manager
        self.corpus = corpus_manager.corpus
        self.training_set = corpus_manager.training_set
        self.test_set = corpus_manager.test_set
        self.all_words = corpus_manager.all_words

        # Extract unique characters
        unique_characters = corpus_manager.extract_unique_characters()
        self.unique_character_count = len(unique_characters)

        # Initialize counts for recall calculation
        self.actual_missing_letter_occurrences = {char: 0 for char in unique_characters}
        self.correct_top_predictions = {char: 0 for char in unique_characters}

        # Initialize counts for precision calculation
        self.top_predicted_counts = {char: 0 for char in unique_characters}
        self.top_correct_retrievals = {char: 0 for char in unique_characters}

        # Initialize prediction class with models, q-gram range, and unique characters
        self.q_range = range(self.config.q_range[0], self.config.q_range[1] + 1)
        self.predictor = Predictions(self.model, self.q_range, unique_characters)

        # Log initialization information using the config from corpus_manager
        logging.info(f'Language Model for {self.corpus_name} initialized with:')
        logging.info(f'Seed: {self.config.seed}')
        logging.info(f'Q-gram Range: {self.config.q_range}')
        logging.info(f'Train-Test Split Configuration: {self.config.split_config}')
        logging.info(f'Training Set Size: {len(corpus_manager.training_set)}')
        logging.info(f'Testing Set Size: {len(corpus_manager.test_set)}')
        logging.info(f'Vowel Replacement Ratio: {self.config.vowel_replacement_ratio}')
        logging.info(f'Consonant Replacement Ratio: {self.config.consonant_replacement_ratio}')
        logging.info(f'Unique Character Count: {self.unique_character_count}')
        logging.info(f'Minimum Word Length: {self.config.min_word_length}')

        # Retrieve the prediction method based on the name
        prediction_methods = {
            'context_sensitive': self.predictor.context_sensitive,
            'context_no_boundary': self.predictor.context_no_boundary,
            'base_prediction': self.predictor.base_prediction
        }
        self.prediction_method = prediction_methods.get(self.config.prediction_method_name, self.predictor.context_sensitive)

    def compute_accuracy(self, predictions) -> dict:
        # Initialize a dictionary to track accuracy for three ranks: TOP1, TOP2, and TOP3.
        accuracy_counts = {1: 0, 2: 0, 3: 0}  # Ensuring all ranks are initialized.
        total_test_words = len(self.test_set)  # Total number of words in the test set.

        for _, missing_letter, _, all_predictions, _ in predictions:
            # Identify the highest rank (1, 2, or 3) where the correct prediction (missing_letter) is made.
            correct_rank = next((rank for rank, (predicted_letter, _) in enumerate(all_predictions, start=1) if predicted_letter == missing_letter), None)
            
            # If a correct prediction is made, increment accuracy counts for that rank and all higher ranks.
            if correct_rank:
                for rank in range(correct_rank, 4):  # Loop from correct rank to 3.
                    accuracy_counts[rank] += 1  # Increment the count for each relevant rank.

        # Calculate total accuracy for each rank (1, 2, 3) by dividing the number of accurate predictions by total test words.
        total_accuracy = {k: accuracy_counts[k] / total_test_words for k in accuracy_counts}
        return total_accuracy

    def compute_validity(self, predictions) -> dict:
        # Initialize a dictionary to track validity for three ranks: TOP1, TOP2, and TOP3.
        validity_counts = {1: 0, 2: 0, 3: 0}
        total_test_words = len(self.test_set)  # Total number of words in the test set.

        for modified_word, _, _, all_predictions, _ in predictions:
            valid_word_found = False  # Flag to indicate if a valid word has been found.
            for rank, (predicted_letter, _) in enumerate(all_predictions, start=1):
                # If a valid word hasn't been found yet, check the current prediction.
                if not valid_word_found:
                    # Reconstruct the word by replacing the underscore with the predicted letter.
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    # Check if the reconstructed word exists in the corpus (valid word).
                    if reconstructed_word in self.all_words:
                        # If valid, increment validity counts for that rank and all higher ranks.
                        for i in range(rank, 4):  # Loop from current rank to 3.
                            validity_counts[i] += 1
                        valid_word_found = True  # Update flag since a valid word is found.

        # Calculate total validity for each rank (1, 2, 3) by dividing the number of valid predictions by total test words.
        total_validity = {k: validity_counts[k] / total_test_words for k in validity_counts}
        return total_validity

    def compute_recall(self) -> dict:
        # Calculate recall for each character in the corpus
        recall_metrics = {
            # Iterate over each character that was missing and needed prediction
            char: (
                # Recall is calculated as the number of times the character was correctly predicted as the top choice (True Positives)
                # divided by the total number of times the character was the missing letter (True Positives + False Negatives)
                self.correct_top_predictions[char] / self.actual_missing_letter_occurrences[char]
                if self.actual_missing_letter_occurrences[char] > 0  # Ensure denominator is not zero
                else 0
            )
            for char in self.actual_missing_letter_occurrences
        }
        return recall_metrics

    def compute_precision(self) -> dict:
        # Calculate precision for each character
        precision_metrics = {
            # For each character, count how often it was correctly predicted as the top choice
            char: (
                # Precision is calculated as the number of correct top predictions (True Positives)
                # divided by the total number of top predictions (True Positives + False Positives)
                self.correct_top_predictions[char] / self.top_predicted_counts[char]
                if self.top_predicted_counts[char] > 0  # Ensure denominator is not zero
                else 0
            )
            for char in self.top_predicted_counts
        }
        return precision_metrics

    def save_recall_precision_stats(self):
        # Calculate and sort recall and precision metrics
        recall_metrics = self.compute_recall()
        precision_metrics = self.compute_precision()

        # Sort metrics by Total Relevant (Actual Missing Letter Occurrences) in descending order
        sorted_metrics = sorted(
            [
                (char, 
                 self.actual_missing_letter_occurrences[char], 
                 self.correct_top_predictions[char], 
                 recall_metrics[char], 
                 precision_metrics[char]
                ) for char in recall_metrics
            ], 
            key=lambda item: item[1], reverse=True
        )

        # Save sorted metrics to a file
        metrics_file_path = self.config.csv_dir / f'{self.corpus_name}_recall_precision_metrics.csv'
        with metrics_file_path.open('w', encoding='utf-8') as file:
            file.write('Character, Total_Missing Letter_Occurrences, Total_Correctly_Retrieved, Recall, Precision\n')
            for char, total_relevant, correctly_retrieved, recall, precision in sorted_metrics:
                file.write(f'{char}, {total_relevant}, {correctly_retrieved}, {recall:.4f}, {precision:.4f}\n')

    def evaluate_character_predictions(self, prediction_method) -> tuple[dict, list]:
        predictions = []

        for modified_word, target_letter, original_word in self.test_set:
            # Update counts for recall
            self.actual_missing_letter_occurrences[target_letter] += 1

            all_predictions = prediction_method(modified_word)

            # Update counts for precision
            if all_predictions:
                top_predicted_char = all_predictions[0][0]
                self.top_predicted_counts[top_predicted_char] += 1
                if top_predicted_char == target_letter:
                    self.top_correct_retrievals[target_letter] += 1

            # Update correct retrievals for recall calculation
            if all_predictions and all_predictions[0][0] == target_letter:
                self.correct_top_predictions[target_letter] += 1  # Corrected line

            # Other calculations remain the same
            correct_letter_rank = next((rank for rank, (retrieved_letter, _) in enumerate(all_predictions, start=1) 
                                        if retrieved_letter == target_letter), None)
            predictions.append((modified_word, target_letter, original_word, all_predictions[:3], correct_letter_rank))

        # Calculate various metrics
        accuracy_metrics = self.compute_accuracy(predictions)
        validity_metrics = self.compute_validity(predictions)
        recall_metrics = self.compute_recall()
        precision_metrics = self.compute_precision()

        return {'accuracy': accuracy_metrics, 'validity': validity_metrics, 'recall': recall_metrics, 'precision': precision_metrics, 'total_words': len(self.test_set)}, predictions

    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
        # Adjust file name to include test-train split and q-gram range
        csv_file_path = self.config.csv_dir / f'{self.corpus_name}_{prediction_method_name}_split{self.config.split_config}_qrange{self.config.q_range[0]}-{self.config.q_range[1]}_prediction.csv'

        with csv_file_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Reordered columns
            writer.writerow(['Tested_Word', 'Original_Word', 'Missing_Letter_Position', 'Word_Length', 
                            'Correct_Letter', 'Predicted_Letter', 'Prediction_Rank', 'Confidence', 
                            'Correct_Letter Rank', 'Is_Valid', 'Is_Accurate'])
            
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
        output_file_path = self.config.text_dir / f'{self.corpus_name}_predictions.txt'

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
            file.write(f'Vowel Ratio: {self.config.vowel_replacement_ratio}, Consonant Ratio: {self.config.consonant_replacement_ratio}\n\n')

            # Detailed prediction results
            for modified_word, missing_letter, original_word, top_three_predictions, correct_letter_rank in predictions:
                file.write(f'Tested Word: {modified_word}, Original Word: {original_word}, Correct Letter: {missing_letter}\n')
                file.write(f'Rank of Correct Letter: {correct_letter_rank}\n')

                for rank, (predicted_letter, confidence) in enumerate(top_three_predictions, start=1):
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    is_valid_word = reconstructed_word in self.all_words

                    file.write(f"Rank {rank}: '{predicted_letter}' (Confidence: {confidence:.8f}), Valid: {is_valid_word}\n")
                
                file.write('\n')

def run(corpus_name, config):
    # Correctly use the static method from CorpusManager
    formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
    logging.info(f'Processing {formatted_corpus_name} Corpus')
    logging.info('-' * 40)

    # Create an instance of CorpusManager
    corpus_manager = CorpusManager(formatted_corpus_name, config)
    CorpusManager.add_to_global_corpus(corpus_manager.corpus) 

    # Initialize EvaluateModel with the corpus manager
    eval_model = EvaluateModel(corpus_manager)

    # Retrieve the prediction method from the Predictions object
    prediction_method = getattr(eval_model.predictor, config.prediction_method_name)
    evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)
    logging.info(f'Evaluated with: {prediction_method.__name__}')

    # Log the accuracy and validity results
    accuracy = evaluation_metrics['accuracy']
    validity = evaluation_metrics['validity']
    logging.info(f'Model evaluation completed for: {corpus_name}')
    logging.info(f'TOP1 ACCURACY: {accuracy[1]:.2%} | TOP1 VALIDITY: {validity[1]:.2%}')
    logging.info(f'TOP2 ACCURACY: {accuracy[2]:.2%} | TOP2 VALIDITY: {validity[2]:.2%}')
    logging.info(f'TOP3 ACCURACY: {accuracy[3]:.2%} | TOP3 VALIDITY: {validity[3]:.2%}')

    # Save the predictions to CSV and text files
    eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
    eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)
    eval_model.save_recall_precision_stats()

    logging.info('-' * 40)

def main():
    config = Config()
    config.setup_logging()
    config.create_directories()

    corpora = ['cmudict', 'brown', 'CLMET3.txt', 'reuters', 'gutenberg', 'inaugural']
    for corpus_name in corpora:
        run(corpus_name, config)

    # Create and evaluate the mega-corpus
    mega_corpus_name = 'mega_corpus'
    with open(config.corpus_dir / f'{mega_corpus_name}.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(CorpusManager.unique_words_all_corpora))

    run(mega_corpus_name, config)

if __name__ == '__main__':
    main()
