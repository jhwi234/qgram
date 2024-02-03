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
        self.corpora = ['brown', 'CLMET3', 'cmudict', 'gutenberg', 'inaugural', 'mega_corpus', 'reuters']
        self.sets_dir = self.base_dir / 'data' / 'outputs' / 'sets'
        self.qgrams_dir = self.base_dir / 'data' / 'outputs' / 'qgrams'
        self.csv_dir = self.base_dir / 'data' / 'outputs' / 'csv'
        self.qgrams_dir.mkdir(exist_ok=True)

    def get_file_path(self, corpus_name, file_type, dataset_type=None):
        if file_type == 'csv':
            return self.csv_dir / f'{corpus_name}_context_sensitive_split0.5_qrange6-6_prediction.csv'
        elif file_type == 'qgram':
            suffix = '_correct_predictions_qgrams.txt' if dataset_type == 'correct' else f'_qgram_{dataset_type}.txt'
            return self.qgrams_dir / f'{corpus_name}{suffix}'

def setup_logging():
    log_file = Path(__file__).resolve().parent / 'corpus_analysis.log'
    logging.basicConfig(level=logging.INFO, format='%(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def generate_qgrams(word, qgram_range):
    return [word[i:i+size] for size in range(qgram_range[0], min(qgram_range[1], len(word)) + 1) 
            for i in range(len(word) - size + 1)]

def read_words_from_file(file_path):
    with file_path.open() as file:
        return {line.strip() for line in file}

def write_qgram_frequencies_to_file(qgram_freq, file_path):
    sorted_qgrams = sorted(qgram_freq.items(), key=lambda item: item[1], reverse=True)
    with file_path.open('w') as file:
        file.writelines(f"{qgram}: {freq}\n" for qgram, freq in sorted_qgrams)

def read_qgram_frequencies(file_path):
    with file_path.open() as file:
        return {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in file}

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

# Metric calculation functions
def calculate_emd(freq1, freq2):
    qgrams = set(freq1.keys()).union(freq2.keys())
    dist1 = [freq1.get(qgram, 0) for qgram in qgrams]
    dist2 = [freq2.get(qgram, 0) for qgram in qgrams]
    return wasserstein_distance(dist1, dist2)

def calculate_frequency_similarity(dict1, dict2):
    norm_dict1 = normalize_frequencies(dict1)
    norm_dict2 = normalize_frequencies(dict2)
    freq_diff = sum(abs(norm_dict1.get(qgram, 0) - norm_dict2.get(qgram, 0)) for qgram in set(norm_dict1).union(norm_dict2))
    return 1 - freq_diff / 2

def jaccard_similarity(freq1, freq2):
    keys = set(freq1.keys()).union(freq2.keys())
    vector1 = [1 if freq1.get(k, 0) > 0 else 0 for k in keys]
    vector2 = [1 if freq2.get(k, 0) > 0 else 0 for k in keys]
    return jaccard_score(vector1, vector2, average='macro')

def spearman_correlation(ranks1, ranks2):
    """Calculate Spearman correlation between two sets of ranks."""
    common_words = ranks1.keys() & ranks2.keys()
    rank_list1 = [ranks1[word]['rank'] for word in common_words]
    rank_list2 = [ranks2[word]['rank'] for word in common_words]
    correlation, _ = spearmanr(rank_list1, rank_list2)
    return correlation

def cosine_similarity(freqs1, freqs2):
    """Calculate cosine similarity between two frequency vectors."""
    # Convert to vectors based on common words
    common_words = set(freqs1.keys()) & set(freqs2.keys())
    vec1 = np.array([freqs1[word]['freq'] for word in common_words])
    vec2 = np.array([freqs2[word]['freq'] for word in common_words])
    return 1 - cosine(vec1, vec2)

def kl_divergence(freq1, freq2):
    keys = set(freq1.keys()).union(freq2.keys())
    vector1 = np.array([max(freq1.get(k, 0), 1e-10) for k in keys])
    vector2 = np.array([max(freq2.get(k, 0), 1e-10) for k in keys])
    return entropy(vector1, vector2)

def overlap_coefficient(freq1, freq2):
    intersection = sum(min(freq1[k], freq2[k]) for k in freq1 if k in freq2)
    smaller_sum = min(sum(freq1.values()), sum(freq2.values()))
    return intersection / smaller_sum if smaller_sum > 0 else 0

def calculate_intersection_count(freq1, freq2):
    intersection_set = set(freq1.keys()).intersection(set(freq2.keys()))
    intersection_count = len(intersection_set)
    smaller_set_size = min(len(freq1), len(freq2))
    return intersection_count / smaller_set_size if smaller_set_size > 0 else 0

def dice_coefficient(freq1, freq2):
    intersection = sum(min(freq1[k], freq2[k]) for k in freq1 if k in freq2)
    total = sum(freq1.values()) + sum(freq2.values())
    return 2 * intersection / total if total > 0 else 0

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

def process_and_save_qgrams(word_set, qgram_file_path, config):
    qgrams = Counter(qgram for word in word_set for qgram in generate_qgrams(word, config.qgram_range))
    write_qgram_frequencies_to_file(qgrams, qgram_file_path)

def process_corpus(corpus_name, config):
    logger = logging.getLogger(corpus_name)
    logger.info(f'Processing corpus: {corpus_name}')
    try:
        # Load corpus data
        train_set, test_set = load_corpus_data(corpus_name, config)

        # File paths for Qgram datasets
        train_file = config.get_file_path(corpus_name, 'qgram', 'train')
        test_file = config.get_file_path(corpus_name, 'qgram', 'test')
        pred_file = config.get_file_path(corpus_name, 'qgram', 'correct')
        csv_file = config.get_file_path(corpus_name, 'csv')
        incorrect_pred_file = config.get_file_path(corpus_name, 'qgram', 'incorrect_pred')

        # Process and save Qgrams for train and test sets
        process_and_save_qgrams(train_set, train_file, config)
        process_and_save_qgrams(test_set, test_file, config)

        # Extract and process Qgrams for correct and incorrect predictions
        correct_qgrams = extract_correct_predictions(csv_file, config.qgram_range)
        incorrect_pred_qgrams = extract_incorrect_predictions(csv_file, config.qgram_range)

        # Write Qgram frequencies to file
        write_qgram_frequencies_to_file(correct_qgrams, pred_file)
        write_qgram_frequencies_to_file(incorrect_pred_qgrams, incorrect_pred_file)

        # Read Qgram frequencies
        train_qgrams = read_qgram_frequencies(train_file)
        test_qgrams = read_qgram_frequencies(test_file)
        pred_qgrams = read_qgram_frequencies(pred_file)
        incorrect_pred_qgrams = read_qgram_frequencies(incorrect_pred_file)

        # Augment with ranks
        train_qgrams_with_ranks = augment_with_ranks(train_qgrams)
        test_qgrams_with_ranks = augment_with_ranks(test_qgrams)
        pred_qgrams_with_ranks = augment_with_ranks(pred_qgrams)
        incorrect_pred_qgrams_with_ranks = augment_with_ranks(incorrect_pred_qgrams)

        # Calculate metrics
        metrics = {
            "EMD (Train-Prediction)": calculate_emd(train_qgrams, pred_qgrams),
            "EMD (Train-Test)": calculate_emd(train_qgrams, test_qgrams),
            "EMD (Train-Incorrect Prediction)": calculate_emd(train_qgrams, incorrect_pred_qgrams),
            "Frequency Similarity (Train-Prediction)": calculate_frequency_similarity(train_qgrams, pred_qgrams),
            "Frequency Similarity (Train-Test)": calculate_frequency_similarity(train_qgrams, test_qgrams),
            "Frequency Similarity (Train-Incorrect Prediction)": calculate_frequency_similarity(train_qgrams, incorrect_pred_qgrams),
            "Jaccard Similarity (Train-Prediction)": jaccard_similarity(train_qgrams, pred_qgrams),
            "Jaccard Similarity (Train-Test)": jaccard_similarity(train_qgrams, test_qgrams),
            "Jaccard Similarity (Train-Incorrect Prediction)": jaccard_similarity(train_qgrams, incorrect_pred_qgrams),
            "Cosine Similarity (Train-Prediction)": cosine_similarity(train_qgrams, pred_qgrams),
            "Cosine Similarity (Train-Test)": cosine_similarity(train_qgrams, test_qgrams),
            "Cosine Similarity (Train-Incorrect Prediction)": cosine_similarity(train_qgrams, incorrect_pred_qgrams),
            "Intersection Count (Train-Prediction)": calculate_intersection_count(train_qgrams, pred_qgrams),
            "Intersection Count (Train-Test)": calculate_intersection_count(train_qgrams, test_qgrams),
            "Intersection Count (Train-Incorrect Prediction)": calculate_intersection_count(train_qgrams, incorrect_pred_qgrams),
            
            "Spearman Correlation (Train-Prediction)": spearman_correlation(train_qgrams_with_ranks, pred_qgrams_with_ranks),
            "Spearman Correlation (Train-Test)": spearman_correlation(train_qgrams_with_ranks, test_qgrams_with_ranks),
            "Spearman Correlation (Train-Incorrect Prediction)": spearman_correlation(train_qgrams_with_ranks, incorrect_pred_qgrams_with_ranks),
            
            "KL Divergence (Train-Prediction)": kl_divergence(train_qgrams, pred_qgrams),
            "KL Divergence (Train-Test)": kl_divergence(train_qgrams, test_qgrams),
            "KL Divergence (Train-Incorrect Prediction)": kl_divergence(train_qgrams, incorrect_pred_qgrams),
            "Overlap Coefficient (Train-Prediction)": overlap_coefficient(train_qgrams, pred_qgrams),
            "Overlap Coefficient (Train-Test)": overlap_coefficient(train_qgrams, test_qgrams),
            "Overlap Coefficient (Train-Incorrect Prediction)": overlap_coefficient(train_qgrams, incorrect_pred_qgrams),
            "Dice Coefficient (Train-Prediction)": dice_coefficient(train_qgrams, pred_qgrams),
            "Dice Coefficient (Train-Test)": dice_coefficient(train_qgrams, test_qgrams),
            "Dice Coefficient (Train-Incorrect Prediction)": dice_coefficient(train_qgrams, incorrect_pred_qgrams)
        }

        # Log results
        log_results(corpus_name, metrics, logger)

        # Control comparisons for mega_corpus
        if corpus_name == 'mega_corpus':
            perform_control_comparisons(corpus_name, config, test_qgrams, logger)

    except Exception as e:
        logger.error(f"Error processing {corpus_name}: {e}")

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
        for metric in metrics_list:
            # Keys for each comparison
            train_pred_key = f"{metric} (Train-Prediction)"
            train_test_key = f"{metric} (Train-Test)"
            train_incorrect_pred_key = f"{metric} (Train-Incorrect Prediction)" 

            # Values for each comparison
            train_pred_value = metrics.get(train_pred_key, 'N/A')
            train_test_value = metrics.get(train_test_key, 'N/A')
            train_incorrect_pred_value = metrics.get(train_incorrect_pred_key, 'N/A')

            # Formatting and rounding off the values
            logger.info(f'   {metric}:')
            logger.info(f'      Train-Prediction: {train_pred_value:.4f}' if isinstance(train_pred_value, (float, int)) else f'      Train-Prediction: N/A')
            logger.info(f'      Train-Test: {train_test_value:.4f}' if isinstance(train_test_value, (float, int)) else f'      Train-Test: N/A')
            logger.info(f'      Train-Incorrect Prediction: {train_incorrect_pred_value:.4f}' if isinstance(train_incorrect_pred_value, (float, int)) else f'      Train-Incorrect Prediction: N/A')

    # Control Analysis for mega_corpus
    if corpus_name == 'mega_corpus':
        logger.info(f'\nControl Analysis for {corpus_name}:')
        for key, value in metrics.items():
            if "Control" in key:
                logger.info(f'   {key}: {value:.4f}' if isinstance(value, (float, int)) else f'   {key}: {value}')

    logger.info(separator)

def perform_control_comparisons(corpus_name, config, test_qgrams, logger):
    inaugural_train_file = config.get_file_path('inaugural', 'qgram', 'train')
    inaugural_train_qgrams = read_qgram_frequencies(inaugural_train_file)
    emd_cross = calculate_emd(test_qgrams, inaugural_train_qgrams)
    freq_sim_cross = calculate_frequency_similarity(test_qgrams, inaugural_train_qgrams)
    logger.info(f'Control Analysis for {corpus_name}:')
    logger.info(f'   EMD (Mega Corpus Test vs. Inaugural Train): {emd_cross:.6f}')
    logger.info(f'   Frequency Similarity (Mega Corpus Test vs. Inaugural Train): {freq_sim_cross:.6f}')

def main():
    setup_logging()
    config = Config(Path(__file__).parent, qgram_range=(2, 6))
    for corpus in config.corpora:
        process_corpus(corpus, config)

if __name__ == "__main__":
    main()