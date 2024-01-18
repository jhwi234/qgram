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
from predictions_class import Predictions

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
        self.training_list = []
        self.test_list = []
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
        self.all_words = set(self.training_list)
        self.all_words.update([original_word for _, _, original_word in self.test_list])

        # Generate the formatted training list path and the corresponding models
        formatted_training_list_path = self.config.sets_dir / f'{self.corpus_name}_formatted_training_list.txt'
        self.generate_formatted_corpus(self.training_list, formatted_training_list_path)
        self.generate_models_from_corpus(formatted_training_list_path)

    def _split_type_a(self):
        """
        Split Type A: Iteratively shuffles and divides word types into training and test sets, ensuring no overlap of the word tokesn.
        Aims to balance the cumulative word count in both sets, adhering closely to the configuration ratio.
        Ensures no duplicate word-letter combinations in the test set.
        """
        # Shuffle unique word types for random distribution
        unique_word_types = list(self.corpus.keys())
        self.rng.shuffle(unique_word_types)

        # Calculate total number of words and target count for training set
        total_words = sum(self.corpus.values())
        target_training_words = int(total_words * self.config.split_config)

        # Initialize counters and set for tracking used combinations
        training_words_count = 0
        used_combinations = set()

        for word in unique_word_types:
            word_count = self.corpus[word]

            # If adding this word type keeps cumulative count near target, add to training list
            if training_words_count + word_count <= target_training_words:
                self.training_list.extend([word] * word_count)
                training_words_count += word_count
            else:
                # Otherwise, add to test list, ensuring no duplicate combinations
                for _ in range(word_count):
                    modified_word, missing_letter, _ = self.replace_random_letter(word, used_combinations)
                    self.test_list.append((modified_word, missing_letter, word))
                    used_combinations.add((word, missing_letter))

        # Save lists to files if debug mode is active
        if self.debug:
            self.save_list_to_file(self.training_list, f'{self.corpus_name}_training_list_a.txt')
            self.save_list_to_file(self.test_list, f'{self.corpus_name}_test_list_a.txt')

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
        training_tokens = all_word_tokens[:split_index]
        test_tokens = all_word_tokens[split_index:]

        # Assign tokens to the training list
        self.training_list.extend(training_tokens)

        # Initialize a set to track used word-letter combinations in test set
        used_combinations = set()
        for word in test_tokens:
            modified_word, missing_letter, _ = self.replace_random_letter(word, used_combinations)
            self.test_list.append((modified_word, missing_letter, word))
            used_combinations.add((word, missing_letter))

        # Save lists to files if debug mode is active
        if self.debug:
            self.save_list_to_file(self.training_list, f'{self.corpus_name}_training_list_b.txt')
            self.save_list_to_file(self.test_list, f'{self.corpus_name}_test_list_b.txt')

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
                formatted_training_list_path = self.config.sets_dir / f'{self.corpus_name}_formatted_training_list.txt'
                self.generate_formatted_corpus(self.training_list, formatted_training_list_path)
                self.generate_models_from_corpus(formatted_training_list_path)

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

class EvaluateModel:
    def __init__(self, corpus_manager):
        self.corpus_manager = corpus_manager
        self.corpus_name = corpus_manager.corpus_name 

        self.config = corpus_manager.config
        self.model = corpus_manager.model  # Use the loaded models

        # Access datasets directly from the provided corpus manager
        self.corpus = corpus_manager.corpus
        self.training_list = corpus_manager.training_list
        self.test_list = corpus_manager.test_list
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
        logging.info(f'Training Set Size: {len(corpus_manager.training_list)}')
        logging.info(f'Testing Set Size: {len(corpus_manager.test_list)}')
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
        total_test_words = len(self.test_list)  # Total number of words in the test set.

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
        total_test_words = len(self.test_list)  # Total number of words in the test set.

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

        for modified_word, target_letter, original_word in self.test_list:
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

        return {'accuracy': accuracy_metrics, 'validity': validity_metrics, 'recall': recall_metrics, 'precision': precision_metrics, 'total_words': len(self.test_list)}, predictions

    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
        # Adjust file name to include test-train split and q-gram range
        csv_file_path = self.config.csv_dir / (
            f'{self.corpus_name}_{prediction_method_name}_split'
            f'{self.config.split_config}_qrange{self.config.q_range[0]}-'
            f'{self.config.q_range[1]}_prediction.csv'
        )

        with csv_file_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Adjusted columns to include top three predictions
            writer.writerow([
                'Tested_Word', 'Original_Word', 'Correct_Letter', 
                'Top1_Predicted_Letter', 'Top1_Confidence', 'Top1_Is_Valid', 'Top1_Is_Accurate',
                'Top2_Predicted_Letter', 'Top2_Confidence', 'Top2_Is_Valid', 'Top2_Is_Accurate',
                'Top3_Predicted_Letter', 'Top3_Confidence', 'Top3_Is_Valid', 'Top3_Is_Accurate',
                'Correct_Letter_Rank'
            ])

            for mod_word, miss_letter, orig_word, top_preds, cor_letter_rank in predictions:
                row = [mod_word, orig_word, miss_letter]

                # Process each of the top three predictions
                for predicted_letter, confidence in top_preds:
                    reconstructed_word = mod_word.replace('_', predicted_letter)
                    is_valid = 1 if reconstructed_word in self.all_words else 0
                    is_accurate = 1 if predicted_letter == miss_letter else 0

                    # Append prediction details to the row
                    row.extend([predicted_letter, confidence, is_valid, is_accurate])

                # Append correct letter rank to the row
                row.append(cor_letter_rank)

                # Write the complete row to the CSV
                writer.writerow(row)

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
            file.write(f'Train Size: {len(self.training_list)}, Test Size: {len(self.test_list)}\n')
            file.write(f'Vowel Ratio: {self.config.vowel_replacement_ratio}, '
                    f'Consonant Ratio: {self.config.consonant_replacement_ratio}\n\n')

            # Detailed prediction results
            for mod_word, miss_letter, orig_word, top_preds, cor_letter_rank in predictions:
                file.write(f'Test Word: {mod_word}, Correct Letter: {miss_letter}\n')
                file.write(f'Correct Letter Rank: {cor_letter_rank}\n')

                for rank, (pred_letter, confidence) in enumerate(top_preds, start=1):
                    reconstructed_word = mod_word.replace('_', pred_letter)
                    is_valid_word = reconstructed_word in self.all_words

                    file.write(f"Rank {rank}: '{pred_letter}' (Confidence: {confidence:.8f}), "
                            f"Valid: {is_valid_word}\n")
                
                file.write('\n')

def run(corpus_name, config, split_type):
    # Format the corpus name for consistency
    formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
    logging.info(f'Processing {formatted_corpus_name} Corpus using Split Type {split_type}')
    logging.info('-' * 40)

    # Initialize the corpus manager with specified corpus, configuration, and split type
    corpus_manager = CorpusManager(formatted_corpus_name, config, split_type)
    
    # Add words from the current corpus to the global corpus list
    CorpusManager.add_to_global_corpus(corpus_manager.corpus)

    # Evaluate the model using the corpus manager
    eval_model = EvaluateModel(corpus_manager)
    
    # Get the prediction method from the config and evaluate predictions
    prediction_method = getattr(eval_model.predictor, config.prediction_method_name)
    evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)

    logging.info(f'Evaluated with: {prediction_method.__name__}')

    # Log the accuracy and validity results for the model evaluation
    accuracy = evaluation_metrics['accuracy']
    validity = evaluation_metrics['validity']
    logging.info(f'Model evaluation completed for: {corpus_name}')
    logging.info(f'TOP1 ACCURACY: {accuracy[1]:.2%} | TOP1 VALIDITY: {validity[1]:.2%}')
    logging.info(f'TOP2 ACCURACY: {accuracy[2]:.2%} | TOP2 VALIDITY: {validity[2]:.2%}')
    logging.info(f'TOP3 ACCURACY: {accuracy[3]:.2%} | TOP3 VALIDITY: {validity[3]:.2%}')

    # Save the prediction details to CSV, summary statistics to text files, and recall-precision stats
    eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
    eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)
    eval_model.save_recall_precision_stats()

    logging.info('-' * 40)

def main():
    config = Config()
    config.setup_logging()
    config.create_directories()

    # Process each corpus with both split types A and B
    corpora = ['cmudict', 'brown', 'CLMET3.txt', 'reuters', 'gutenberg', 'inaugural']
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