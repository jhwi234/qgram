import logging
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance, entropy
from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score

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
        if file_type == 'csv':
            return self.csv_dir / f"{corpus_name}_context_sensitive_split0.5_qrange6-6_prediction.csv"
        elif file_type == 'qgram':
            suffix = '_correct_predictions_qgrams.txt' if dataset_type == 'correct' else f'_{dataset_type}_qgrams.txt'
            return self.qgrams_dir / f"{corpus_name}{suffix}"

# Update the setup_logging function to use the log directory
def setup_logging(config):
    log_file = config.log_dir / 'corpus_analysis.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', 
                        handlers=[logging.FileHandler(log_file, 'a', 'utf-8'), logging.StreamHandler()])

def generate_qgrams(word, qgram_range):
    return [word[i:i+size] for size in range(qgram_range[0], qgram_range[1]+1) for i in range(len(word) - size + 1)]

def read_words_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return {line.strip() for line in file}
    except Exception as e:
        logging.error(f"Failed to read from {file_path}: {e}")
        return set()

def write_qgram_frequencies_to_file(qgram_freq, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for qgram, freq in sorted(qgram_freq.items(), key=lambda item: item[1], reverse=True):
                file.write(f"{qgram}: {freq}\n")
    except Exception as e:
        logging.error(f"Failed to write to {file_path}: {e}")

def read_qgram_frequencies(file_path):
    frequencies = {}
    try:
        with file_path.open('r', encoding='utf-8') as file:
            frequencies = {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in file}
    except Exception as e:
        logging.error(f"Failed to read from {file_path}: {e}")
    return frequencies

# Extract q-grams from correctly predicted words
def extract_correct_predictions(csv_file_path, qgram_range):
    with csv_file_path.open() as file:
        reader = csv.DictReader(file)
        correct_words = [row['Original_Word'] for row in reader if row['Correct_Letter'] == row['Top1_Predicted_Letter']]
        return Counter(qgram for word in correct_words for qgram in generate_qgrams(word, qgram_range))

# Extract q-grams from incorrectly predicted words
def extract_incorrect_predictions(csv_file_path, qgram_range):
    with csv_file_path.open() as file:
        reader = csv.DictReader(file)
        incorrect_words = [row['Original_Word'] for row in reader if row['Correct_Letter'] != row['Top1_Predicted_Letter']]
        return Counter(qgram for word in incorrect_words for qgram in generate_qgrams(word, qgram_range))

def load_corpus_data(corpus_name, config):
    all_words_file = config.sets_dir / f'{corpus_name}_all_words.txt'
    train_set_file = config.sets_dir / f'{corpus_name}_train_set.txt'
    all_words = read_words_from_file(all_words_file)
    train_set = read_words_from_file(train_set_file)
    if not all_words or not train_set:
        raise ValueError(f"Missing data for {corpus_name}")
    return train_set, all_words - train_set

def normalize_frequencies(qgram_freq):
    total = sum(qgram_freq.values())
    return {qgram: freq / total for qgram, freq in qgram_freq.items()}

def augment_with_ranks(qgram_freqs):
    """Augment frequency dictionaries with ranks."""
    sorted_items = sorted(qgram_freqs.items(), key=lambda x: x[1], reverse=True)
    ranked_qgrams = {}
    for rank, (qgram, freq) in enumerate(sorted_items, start=1):
        ranked_qgrams[qgram] = {'freq': freq, 'rank': rank}
    return ranked_qgrams

def process_and_save_qgrams(word_set, qgram_file_path, config):
    qgrams = Counter(qgram for word in word_set for qgram in generate_qgrams(word, config.qgram_range))
    write_qgram_frequencies_to_file(qgrams, qgram_file_path)

def log_metric(metric_name, metric_value):
    formatted_value = format_value(metric_value)  # Use your format_value function
    logging.info(f"{metric_name}: {formatted_value}")

def vectorize_and_normalize(qgrams, qgrams_union):
    """Convert q-gram frequencies to vector and normalize it for Cosine Similarity."""
    vec = np.array([qgrams.get(qgram, 0) for qgram in qgrams_union])
    norm_vec = vec / np.linalg.norm(vec)
    return norm_vec

# Metric calculation functions
def calculate_emd_distance(freq1, freq2):
    if not freq1 or not freq2:  # Check if either distribution is empty
        logging.warning("Attempted to calculate EMD on empty distribution.")
        return None  # Or handle appropriately, e.g., return a default value or raise a custom exception
    qgrams = set(freq1.keys()).union(freq2.keys())
    dist1 = [freq1.get(qgram, 0) for qgram in qgrams]
    dist2 = [freq2.get(qgram, 0) for qgram in qgrams]
    return wasserstein_distance(dist1, dist2)

def calculate_frequency_similarity(dict1, dict2):
    norm_dict1 = normalize_frequencies(dict1)
    norm_dict2 = normalize_frequencies(dict2)
    freq_diff = sum(abs(norm_dict1.get(qgram, 0) - norm_dict2.get(qgram, 0)) for qgram in set(norm_dict1).union(norm_dict2))
    return 1 - freq_diff / 2

def calculate_jaccard_similarity(freq1, freq2):
    keys = set(freq1.keys()).union(freq2.keys())
    vector1 = [1 if freq1.get(k, 0) > 0 else 0 for k in keys]
    vector2 = [1 if freq2.get(k, 0) > 0 else 0 for k in keys]
    return jaccard_score(vector1, vector2, average='macro')

def calculate_spearman_correlation(ranks1, ranks2):
    """Calculate Spearman correlation between two sets of ranks."""
    common_words = ranks1.keys() & ranks2.keys()
    rank_list1 = [ranks1[word]['rank'] for word in common_words]
    rank_list2 = [ranks2[word]['rank'] for word in common_words]
    correlation, _ = spearmanr(rank_list1, rank_list2)
    return correlation

def calculate_cosine_similarity(freqs1, freqs2):
    """Calculate cosine similarity between two frequency vectors."""
    # Convert to vectors based on common words
    common_words = set(freqs1.keys()) & set(freqs2.keys())
    vec1 = np.array([freqs1[word]['freq'] for word in common_words])
    vec2 = np.array([freqs2[word]['freq'] for word in common_words])
    return 1 - cosine(vec1, vec2)

def calculate_kl_divergence(freq1, freq2):
    keys = set(freq1.keys()).union(freq2.keys())
    vector1 = np.array([max(freq1.get(k, 0), 1e-10) for k in keys])
    vector2 = np.array([max(freq2.get(k, 0), 1e-10) for k in keys])
    return entropy(vector1, vector2)

def calculate_overlap_coefficient(freq1, freq2):
    intersection = sum(min(freq1[k], freq2[k]) for k in freq1 if k in freq2)
    smaller_sum = min(sum(freq1.values()), sum(freq2.values()))
    return intersection / smaller_sum if smaller_sum > 0 else 0

def calculate_intersection_count(freq1, freq2):
    intersection_set = set(freq1.keys()).intersection(set(freq2.keys()))
    intersection_count = len(intersection_set)
    smaller_set_size = min(len(freq1), len(freq2))
    return intersection_count / smaller_set_size if smaller_set_size > 0 else 0

def calculate_dice_coefficient(freq1, freq2):
    intersection = sum(min(freq1[k], freq2[k]) for k in freq1 if k in freq2)
    total = sum(freq1.values()) + sum(freq2.values())
    return 2 * intersection / total if total > 0 else 0

def calculate_metrics(train_qgrams, test_qgrams, pred_qgrams, incorrect_pred_qgrams):
    """Calculate various metrics between q-gram frequency vectors."""
    qgrams_union = set(train_qgrams) | set(test_qgrams) | set(pred_qgrams) | set(incorrect_pred_qgrams)
    
    train_vec = np.array([train_qgrams.get(qgram, 0) for qgram in qgrams_union])
    test_vec = np.array([test_qgrams.get(qgram, 0) for qgram in qgrams_union])
    pred_vec = np.array([pred_qgrams.get(qgram, 0) for qgram in qgrams_union])
    incorrect_vec = np.array([incorrect_pred_qgrams.get(qgram, 0) for qgram in qgrams_union])

    metrics = {
        "EMD Train-Pred": calculate_emd_distance(train_vec, pred_vec),
        "EMD Train-Test": calculate_emd_distance(train_vec, test_vec),
        "EMD Train-Incorrect Pred": calculate_emd_distance(train_vec, incorrect_vec),
    
        "Cosine Similarity Train-Pred": calculate_cosine_similarity(train_vec, pred_vec),
        "Cosine Similarity Train-Test": calculate_cosine_similarity(train_vec, test_vec),
        "Cosine Similarity Train-Incorrect Pred": calculate_cosine_similarity(train_vec, incorrect_vec),
        
        "Spearman Correlation Train-Pred": calculate_spearman_correlation(train_vec, pred_vec),
        "Spearman Correlation Train-Test": calculate_spearman_correlation(train_vec, test_vec),
        "Spearman Correlation Train-Incorrect Pred": calculate_spearman_correlation(train_vec, incorrect_vec),
        
        "KL Divergence Train-Pred": calculate_kl_divergence(train_vec, pred_vec),
        "KL Divergence Train-Test": calculate_kl_divergence(train_vec, test_vec),
        "KL Divergence Train-Incorrect Pred": calculate_kl_divergence(train_vec, incorrect_vec),
        
        "Jaccard Similarity Train-Pred": calculate_jaccard_similarity(train_qgrams, pred_qgrams),
        "Jaccard Similarity Train-Test": calculate_jaccard_similarity(train_qgrams, test_qgrams),
        "Jaccard Similarity Train-Incorrect Pred": calculate_jaccard_similarity(train_qgrams, incorrect_pred_qgrams),
        
        "Frequency Similarity Train-Pred": calculate_frequency_similarity(train_qgrams, pred_qgrams),
        "Frequency Similarity Train-Test": calculate_frequency_similarity(train_qgrams, test_qgrams),
        "Frequency Similarity Train-Incorrect Pred": calculate_frequency_similarity(train_qgrams, incorrect_pred_qgrams),
        
        "Overlap Coefficient Train-Pred": calculate_overlap_coefficient(train_qgrams, pred_qgrams),
        "Overlap Coefficient Train-Test": calculate_overlap_coefficient(train_qgrams, test_qgrams),
        "Overlap Coefficient Train-Incorrect Pred": calculate_overlap_coefficient(train_qgrams, incorrect_pred_qgrams),
        
        "Dice Coefficient Train-Pred": calculate_dice_coefficient(train_qgrams, pred_qgrams),
        "Dice Coefficient Train-Test": calculate_dice_coefficient(train_qgrams, test_qgrams),
        "Dice Coefficient Train-Incorrect Pred": calculate_dice_coefficient(train_qgrams, incorrect_pred_qgrams),
        
        "Intersection Count Train-Pred": calculate_intersection_count(train_qgrams, pred_qgrams),
        "Intersection Count Train-Test": calculate_intersection_count(train_qgrams, test_qgrams),
        "Intersection Count Train-Incorrect Pred": calculate_intersection_count(train_qgrams, incorrect_pred_qgrams),
    }
    
    return metrics

# Main Processing Functions
def process_corpus(corpus_name, config):
    try:
        # Example placeholders for loading your data
        train_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'train'))
        test_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'test'))
        pred_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'correct'))
        incorrect_pred_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'incorrect_pred'))

        # Calculate metrics
        metrics = calculate_metrics(train_qgrams, test_qgrams, pred_qgrams, incorrect_pred_qgrams)

        # Log metrics
        log_results(corpus_name, metrics)

        logging.info(f"Successfully processed {corpus_name}")
    except Exception as e:
        logging.error(f"Error processing {corpus_name}: {e}")

def format_value(value):
    if value is None:
        return 'N/A'  # or any other placeholder you prefer
    return f'{value:.4f}' if isinstance(value, float) else value

def log_results(corpus_name, metrics, logger):
    separator = '-' * 50
    header = f"{corpus_name} Corpus Analysis"
    logger.info(f'\n{separator}\n{header}\n{separator}')

    # Grouping and formatting the metrics for better readability
    similarity_metrics = ['EMD', 'Frequency Similarity', 'Jaccard Similarity', 'Cosine Similarity']
    statistical_metrics = ['KL Divergence']
    overlap_metrics = ['Overlap Coefficient', 'Dice Coefficient', 'Intersection Count']

    for group, metrics_list in [('Similarity Metrics', similarity_metrics), 
                                ('Statistical Metrics', statistical_metrics), 
                                ('Overlap Metrics', overlap_metrics)]:
        logger.info(f'\n{group}:')
        # Inside log_results function
        for metric in metrics_list:
            # Keys for each comparison
            train_pred_key = f"{metric} (Train-Prediction)"
            train_test_key = f"{metric} (Train-Test)"
            train_incorrect_pred_key = f"{metric} (Train-Incorrect Prediction)"

            # Correct usage of format_value for logging within log_results
            logger.info(f'   {metric}:')
            logger.info(f'      Train-Prediction: {format_value(metrics.get(train_pred_key))}')
            logger.info(f'      Train-Test: {format_value(metrics.get(train_test_key))}')
            logger.info(f'      Train-Incorrect Prediction: {format_value(metrics.get(train_incorrect_pred_key))}')

    # Control Analysis for mega_corpus
    if corpus_name == 'mega_corpus':
        logger.info(f'\nControl Analysis for {corpus_name}:')
        for key, value in metrics.items():
            if "Control" in key:
                logger.info(f'   {key}: {value:.4f}' if isinstance(value, (float, int)) else f'   {key}: {value}')

    logger.info(separator)

def main():
    setup_logging(Config(Path.cwd()))
    config = Config(Path.cwd())
    for corpus in config.corpora:
        process_corpus(corpus, config)

if __name__ == "__main__":
    main()