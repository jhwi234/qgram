from collections import Counter
from pathlib import Path
import logging
import csv
from scipy.stats import wasserstein_distance

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
    log_file = Path(__file__).parent / 'corpus_analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
def generate_qgrams(word, qgram_range):
    return [word[i:i+size] for size in range(qgram_range[0], min(qgram_range[1], len(word)) + 1) for i in range(len(word) - size + 1)]

def read_words_from_file(file_path):
    with file_path.open('r', encoding='utf-8') as file:
        return set(file.read().splitlines())

def write_qgram_frequencies_to_file(qgram_freq, file_path):
    sorted_qgrams = sorted(qgram_freq.items(), key=lambda item: item[1], reverse=True)
    with file_path.open('w', encoding='utf-8') as file:
        file.writelines(f"{qgram}: {freq}\n" for qgram, freq in sorted_qgrams)

def read_qgram_frequencies(file_path):
    with file_path.open('r', encoding='utf-8') as file:
        return {line.split(': ')[0]: int(line.split(': ')[1]) for line in file.readlines()}

def normalize_frequencies(qgram_freq):
    total = sum(qgram_freq.values())
    return {qgram: freq / total for qgram, freq in qgram_freq.items()}

def calculate_emd(freq1, freq2):
    qgrams = set(freq1.keys()).union(freq2.keys())
    dist1 = [freq1.get(qgram, 0) for qgram in qgrams]
    dist2 = [freq2.get(qgram, 0) for qgram in qgrams]
    return wasserstein_distance(dist1, dist2)

def calculate_frequency_similarity(dict1, dict2):
    norm_dict1 = normalize_frequencies(dict1)
    norm_dict2 = normalize_frequencies(dict2)
    total_count = sum(norm_dict1.values()) + sum(norm_dict2.values())
    frequency_difference = sum(abs(norm_dict1.get(qgram, 0) - norm_dict2.get(qgram, 0)) for qgram in set(norm_dict1).union(norm_dict2))
    normalized_difference = frequency_difference / total_count if total_count > 0 else 0
    return 1 - normalized_difference

def extract_correct_predictions(csv_file_path, config):
    with csv_file_path.open('r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        qgrams = Counter(qgram for row in reader if row['Correct_Letter'] == row['Top1_Predicted_Letter']
                         for qgram in generate_qgrams(row['Original_Word'], config.qgram_range))
    return qgrams

def process_and_save_qgrams(word_set, qgram_file_path, config):
    qgrams = Counter(qgram for word in word_set for qgram in generate_qgrams(word, config.qgram_range))
    write_qgram_frequencies_to_file(qgrams, qgram_file_path)

def process_corpus(corpus_name, config):
    logger = logging.getLogger(corpus_name)
    logger.info(f'Processing corpus: {corpus_name}')
    try:
        train_set, test_set = load_corpus_data(corpus_name, config)
        train_file = config.get_file_path(corpus_name, 'qgram', 'train')
        test_file = config.get_file_path(corpus_name, 'qgram', 'test')
        pred_file = config.get_file_path(corpus_name, 'qgram', 'correct')
        csv_file = config.get_file_path(corpus_name, 'csv')

        process_and_save_qgrams(train_set, train_file, config)
        process_and_save_qgrams(test_set, test_file, config)
        correct_qgrams = extract_correct_predictions(csv_file, config)
        write_qgram_frequencies_to_file(correct_qgrams, pred_file)

        train_qgrams = read_qgram_frequencies(train_file)
        pred_qgrams = read_qgram_frequencies(pred_file)
        test_qgrams = read_qgram_frequencies(test_file)

        # Existing analysis
        emd_train_pred = calculate_emd(train_qgrams, pred_qgrams)
        emd_train_test = calculate_emd(train_qgrams, test_qgrams)
        freq_sim_train_pred = calculate_frequency_similarity(train_qgrams, pred_qgrams)
        freq_sim_train_test = calculate_frequency_similarity(train_qgrams, test_qgrams)
        log_results(corpus_name, emd_train_pred, emd_train_test, freq_sim_train_pred, freq_sim_train_test, logger)

        # Perform control comparisons
        if corpus_name == 'mega_corpus':
            perform_control_comparisons(corpus_name, config, test_qgrams)

    except Exception as e:
        logger.error(f"Error processing {corpus_name}: {e}")

def load_corpus_data(corpus_name, config):
    all_words_file = config.sets_dir / f'{corpus_name}_all_words.txt'
    train_set_file = config.sets_dir / f'{corpus_name}_train_set.txt'
    all_words = read_words_from_file(all_words_file)
    train_set = read_words_from_file(train_set_file)
    if not all_words or not train_set:
        raise ValueError(f"Missing data for {corpus_name}")
    return train_set, all_words - train_set

def log_results(corpus_name, emd_train_pred, emd_train_test, freq_sim_train_pred, freq_sim_train_test, logger):
    logger.info(f'{corpus_name} Corpus Analysis:')
    logger.info(f'   EMD (Train-Prediction): {emd_train_pred:.6f}')
    logger.info(f'   EMD (Train-Test): {emd_train_test:.6f}')
    logger.info(f'   Frequency Similarity (Train-Prediction): {freq_sim_train_pred:.6f}')
    logger.info(f'   Frequency Similarity (Train-Test): {freq_sim_train_test:.6f}\n')

def perform_control_comparisons(corpus_name, config, test_qgrams):
    logger = logging.getLogger(corpus_name)

    # Cross-corpus comparison (Mega corpus test vs. Inaugural train)
    if corpus_name == 'mega_corpus':
        inaugural_train_file = config.get_file_path('inaugural', 'qgram', 'train')
        inaugural_train_qgrams = read_qgram_frequencies(inaugural_train_file)
        emd_cross = calculate_emd(test_qgrams, inaugural_train_qgrams)
        freq_sim_cross = calculate_frequency_similarity(test_qgrams, inaugural_train_qgrams)

        # Log control results
        logger.info(f'Control Analysis for {corpus_name}:')
        logger.info(f'   EMD (Mega Corpus Test vs. Inaugural Train): {emd_cross}')
        logger.info(f'   Frequency Similarity (Mega Corpus Test vs. Inaugural Train): {freq_sim_cross}\n')

def main():
    setup_logging()
    config = Config(Path(__file__).parent, qgram_range=(2, 6))
    for corpus in config.corpora:
        process_corpus(corpus, config)

if __name__ == "__main__":
    main()