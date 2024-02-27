import logging
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

class Config:
    def __init__(self, base_dir, qgram_range=(2, 6)):
        self.base_dir = Path(base_dir)
        self.qgram_range = qgram_range
        self.corpora = ['brown', 'CLMET3', 'cmudict', 'gutenberg', 'inaugural', 'reuters']
        self._set_directories()
    
    def _set_directories(self):
        # Setup various directories needed for the application
        self.data_dir = self.base_dir / 'data'
        self.model_dir = self.data_dir / 'models'
        self.corpus_dir = self.data_dir / 'corpora'
        self.log_dir = self.data_dir / 'logs'
        self.output_dir = self.data_dir / 'outputs'
        self.text_dir = self.output_dir / 'texts'
        self.qgrams_dir = self.output_dir / 'qgrams'
        self.csv_dir = self.output_dir / 'csv'
        self.sets_dir = self.output_dir / 'sets'
        
        # Create directories if they don't exist
        for path in [self.data_dir, self.model_dir, self.log_dir, self.corpus_dir, self.output_dir, self.text_dir, self.csv_dir, self.sets_dir, self.qgrams_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, corpus_name, file_type, dataset_type=None):
        # Reference the available q-gram files
        if file_type == 'csv':
            return self.csv_dir / f"{corpus_name}_context_sensitive_split0.5_qrange6-6_prediction.csv"
        elif file_type == 'qgram':
            # Reference the available q-gram files
            suffix = '_train_qgrams.txt' if dataset_type == 'train' else '_test_qgrams.txt'
            return self.qgrams_dir / f"{corpus_name}{suffix}"

# Setup_logging function use the log directory
def setup_logging(config):
    log_file = config.log_dir / 'corpus_analysis.log'
    logging.basicConfig(level=logging.INFO, format='%(message)s', 
                        handlers=[logging.FileHandler(log_file, 'a', 'utf-8'), logging.StreamHandler()])

def generate_qgrams(word, qgram_range):
    """
    Generate q-grams for a given word.
    """
    return [word[i:i+size] for size in range(qgram_range[0], qgram_range[1]+1) for i in range(len(word) - size + 1)]

def read_words_from_file(file_path):
    """
    Read words from a file and return a set of distinct words.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return {line.strip() for line in file}
    except Exception as e:
        logging.error(f"Failed to read from {file_path}: {e}")
        return set()

def write_qgram_frequencies_to_file(qgram_freq, file_path):
    """
    Write q-gram frequencies to a file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for qgram, freq in sorted(qgram_freq.items(), key=lambda item: item[1], reverse=True):
                file.write(f"{qgram}: {freq}\n")
    except Exception as e:
        logging.error(f"Failed to write to {file_path}: {e}")

def read_qgram_frequencies(file_path):
    """
    Read q-gram frequencies from a file and return a dictionary.
    """
    frequencies = {}
    try:
        with file_path.open('r', encoding='utf-8') as file:
            frequencies = {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in file}
    except Exception as e:
        logging.error(f"Failed to read from {file_path}: {e}")
    return frequencies

# Extract q-grams from correctly predicted words for q-gram extraction
def extract_correct_predictions(csv_file_path, qgram_range):
    with csv_file_path.open() as file:
        reader = csv.DictReader(file)
        correct_words = [row['Original_Word'] for row in reader if row['Correct_Letter'] == row['Top1_Predicted_Letter']]
        return Counter(qgram for word in correct_words for qgram in generate_qgrams(word, qgram_range))

# Extract q-grams from incorrectly predicted words for q-gram extraction
def extract_incorrect_predictions(csv_file_path, qgram_range):
    with csv_file_path.open() as file:
        reader = csv.DictReader(file)
        incorrect_words = [row['Original_Word'] for row in reader if row['Correct_Letter'] != row['Top1_Predicted_Letter']]
        return Counter(qgram for word in incorrect_words for qgram in generate_qgrams(word, qgram_range))

# Load all words and training words to be used for q-gram extraction
def load_corpus_data(corpus_name, config):
    all_words_file = config.sets_dir / f'{corpus_name}_all_words.txt'
    train_set_file = config.sets_dir / f'{corpus_name}_train_set.txt'
    all_words = read_words_from_file(all_words_file)
    train_set = read_words_from_file(train_set_file)
    if not all_words or not train_set:
        raise ValueError(f"Missing data for {corpus_name}")
    return train_set, all_words - train_set

def normalize_frequencies(arr):
    # If 'arr' is actually a dictionary, then we need to sum its values instead
    if isinstance(arr, dict):
        total = sum(arr.values())
        return {k: v / total for k, v in arr.items()} if total > 0 else {k: 0 for k in arr.keys()}
    else:
        # Assuming 'arr' is a NumPy array or similar, for non-dict scenarios
        total = arr.sum()
        return arr / total if total > 0 else np.zeros(arr.shape)

def vectorize_and_normalize(frequencies, qgrams_union):
    vec = np.array([frequencies[qgram] if qgram in frequencies else 0 for qgram in qgrams_union])
    return normalize_frequencies(vec)

def process_and_save_qgrams(word_set, qgram_file_path, config):
    qgrams = Counter(qgram for word in word_set for qgram in generate_qgrams(word, config.qgram_range))
    write_qgram_frequencies_to_file(qgrams, qgram_file_path)

def log_metric(metric_name, metric_value):
    formatted_value = format_value(metric_value)  # Use your format_value function
    logging.info(f"{metric_name}: {formatted_value}")

def calculate_frequency_similarity(arr1, arr2):
    # Assuming arr1 and arr2 are NumPy arrays and not dictionaries
    norm_arr1 = normalize_frequencies(arr1)
    norm_arr2 = normalize_frequencies(arr2)
    freq_diff = np.sum(np.abs(norm_arr1 - norm_arr2))
    return 1 - freq_diff / 2

def calculate_spearman_correlation(arr1, arr2):
    # Assuming arr1 and arr2 are aligned and directly comparable
    correlation, _ = spearmanr(arr1, arr2)
    return correlation if not np.isnan(correlation) else 0

def calculate_intersection_count(arr1, arr2):
    # Assuming arr1 and arr2 are boolean arrays indicating the presence of q-grams
    intersection = np.logical_and(arr1, arr2).sum()
    smaller_set_size = min(arr1.sum(), arr2.sum())
    return intersection / smaller_set_size if smaller_set_size > 0 else 0

def calculate_dice_coefficient(arr1, arr2):
    # Assuming arr1 and arr2 are frequency arrays
    intersection = np.minimum(arr1, arr2).sum()
    total = arr1.sum() + arr2.sum()
    return 2 * intersection / total if total > 0 else 0

def calculate_metrics(train_qgrams, test_qgrams, pred_qgrams, incorrect_pred_qgrams):
    """Calculate various metrics between q-gram frequency vectors."""
    qgrams_union = set(train_qgrams) | set(test_qgrams) | set(pred_qgrams) | set(incorrect_pred_qgrams)

    train_vec = np.array([train_qgrams.get(qgram, 0) for qgram in qgrams_union])
    test_vec = np.array([test_qgrams.get(qgram, 0) for qgram in qgrams_union])
    pred_vec = np.array([pred_qgrams.get(qgram, 0) for qgram in qgrams_union])
    incorrect_vec = np.array([incorrect_pred_qgrams.get(qgram, 0) for qgram in qgrams_union])

    metrics = {
        "Spearman Correlation Train-Correct Pred": calculate_spearman_correlation(train_vec, pred_vec),
        "Spearman Correlation Train-Test": calculate_spearman_correlation(train_vec, test_vec),
        "Spearman Correlation Train-Incorrect Pred": calculate_spearman_correlation(train_vec, incorrect_vec),
        
        "Frequency Similarity Train-Correct Pred": calculate_frequency_similarity(train_vec, pred_vec),
        "Frequency Similarity Train-Test": calculate_frequency_similarity(train_vec, test_vec),
        "Frequency Similarity Train-Incorrect Pred": calculate_frequency_similarity(train_vec, incorrect_vec),
        
        "Dice Coefficient Train-Correct Pred": calculate_dice_coefficient(train_vec, pred_vec),
        "Dice Coefficient Train-Test": calculate_dice_coefficient(train_vec, test_vec),
        "Dice Coefficient Train-Incorrect Pred": calculate_dice_coefficient(train_vec, incorrect_vec),
        
        "Intersection Count Train-Correct Pred": calculate_intersection_count(train_vec, pred_vec),
        "Intersection Count Train-Test": calculate_intersection_count(train_vec, test_vec),
        "Intersection Count Train-Incorrect Pred": calculate_intersection_count(train_vec, incorrect_vec),
    }
    
    return metrics

# Main Processing Functions
def process_corpus(corpus_name, config):
    try:
        # Load q-grams from correctly and incorrectly predicted words
        correct_qgrams = extract_correct_predictions(config.get_file_path(corpus_name, 'csv', 'correct'), config.qgram_range)
        incorrect_qgrams = extract_incorrect_predictions(config.get_file_path(corpus_name, 'csv', 'incorrect_pred'), config.qgram_range)

        # Save correct and incorrect q-grams to files
        correct_qgram_file_path = config.qgrams_dir / f"{corpus_name}_correct_qgrams.txt"
        incorrect_qgram_file_path = config.qgrams_dir / f"{corpus_name}_incorrect_qgrams.txt"
        
        write_qgram_frequencies_to_file(correct_qgrams, correct_qgram_file_path)
        write_qgram_frequencies_to_file(incorrect_qgrams, incorrect_qgram_file_path)

        # Proceed with existing logic to calculate and log metrics
        train_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'train'))
        test_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'test'))
        
        # Note: Assuming 'pred_qgrams' and 'incorrect_pred_qgrams' should be derived from the saved files
        pred_qgrams = read_qgram_frequencies(correct_qgram_file_path)
        incorrect_pred_qgrams = read_qgram_frequencies(incorrect_qgram_file_path)

        metrics = calculate_metrics(train_qgrams, test_qgrams, pred_qgrams, incorrect_pred_qgrams)

        log_results(corpus_name, metrics)

    except Exception as e:
        logging.error(f"Error processing {corpus_name}: {e}")

def format_value(value):
    if value is None:
        return 'N/A'  # or any other placeholder you prefer
    return f'{value:.4f}' if isinstance(value, float) else value

def log_results(corpus_name, metrics):
    separator = '-' * 50
    header = f"{corpus_name} Corpus Analysis"
    logging.info(f'\n{separator}\n{header}\n{separator}')

    last_category = None
    for metric_name in sorted(metrics.keys()):
        # Extract the category from the metric name (assuming category is the prefix up to the first space)
        current_category = metric_name.split(" ")[0]

        # Check if we've moved to a new category of metrics based on prefix change
        if last_category is not None and current_category != last_category:
            logging.info("")  # Add a line break for readability between categories

        # Log the metric
        logging.info(f"{metric_name}: {format_value(metrics[metric_name])}")
        
        # Update the last category tracker
        last_category = current_category

def main():
    setup_logging(Config(Path.cwd()))
    config = Config(Path.cwd())
    for corpus in config.corpora:
        process_corpus(corpus, config)

if __name__ == "__main__":
    main()