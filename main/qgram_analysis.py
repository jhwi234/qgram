import logging
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from corpus_class import CorpusManager
from evaluation_class import EvaluateModel
import nltk
import os

class Config:
    def __init__(self, base_dir, qgram_range=(2, 7)):
        self.base_dir = Path(base_dir)
        self.q_range = list(range(qgram_range[0], qgram_range[1] + 1))
        self.corpora = ['brown', 'CLMET3', 'cmudict', 'sorted_tokens_lampeter', 'sorted_tokens_openEdges']
        self.seed = 42
        self.split_config = 0.5
        self.vowel_replacement_ratio = 0.2
        self.consonant_replacement_ratio = 0.8
        self.min_word_length = 3
        self.prediction_method_name = 'context_sensitive'
        self.num_replacements = 1
        self._set_directories()

    def _set_directories(self):
        self.data_dir = self.base_dir / 'data'
        self.model_dir = self.data_dir / 'models'
        self.corpus_dir = self.data_dir / 'corpora'
        self.log_dir = self.data_dir / 'logs'
        self.output_dir = self.data_dir / 'outputs'
        self.text_dir = self.output_dir / 'texts'
        self.qgrams_dir = self.output_dir / 'qgrams'
        self.csv_dir = self.output_dir / 'csv'
        self.sets_dir = self.output_dir / 'sets'
        
        for path in [self.data_dir, self.model_dir, self.log_dir, self.corpus_dir, self.output_dir, self.text_dir, self.csv_dir, self.sets_dir, self.qgrams_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, corpus_name, file_type, dataset_type=None):
        if file_type == 'csv':
            return self.csv_dir / f"{corpus_name}_context_sensitive_split{self.split_config}_qrange{self.q_range[0]}-{self.q_range[-1]}_prediction.csv"
        elif file_type == 'qgram':
            suffix = '_train_qgrams.txt' if dataset_type == 'train' else '_test_qgrams.txt'
            return self.qgrams_dir / f"{corpus_name}{suffix}"

def setup_logging(config):
    log_file = config.log_dir / 'corpus_analysis.log'
    logging.basicConfig(level=logging.INFO, format='%(message)s', 
                        handlers=[logging.FileHandler(log_file, 'a', 'utf-8'), logging.StreamHandler()])

def generate_qgrams(word, q_range):
    return [word[i:i+size] for size in q_range for i in range(len(word) - size + 1)]

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
    try:
        with file_path.open('r', encoding='utf-8') as file:
            return {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in file}
    except Exception as e:
        logging.error(f"Failed to read from {file_path}: {e}")
        return {}

def normalize_frequencies(arr):
    total = sum(arr.values()) if isinstance(arr, dict) else arr.sum()
    if total == 0:
        return {k: 0 for k in arr.keys()} if isinstance(arr, dict) else np.zeros(arr.shape)
    return {k: v / total for k, v in arr.items()} if isinstance(arr, dict) else arr / total

def process_and_save_qgrams(word_set, qgram_file_path, config):
    qgrams = Counter(qgram for word in word_set for qgram in generate_qgrams(word, config.q_range))
    write_qgram_frequencies_to_file(qgrams, qgram_file_path)

def calculate_frequency_similarity(arr1, arr2):
    norm_arr1, norm_arr2 = map(normalize_frequencies, (arr1, arr2))
    return 1 - np.sum(np.abs(norm_arr1 - norm_arr2)) / 2

def calculate_intersection_count(arr1, arr2):
    intersection = np.logical_and(arr1, arr2).sum()
    smaller_set_size = min(arr1.sum(), arr2.sum())
    return intersection / smaller_set_size if smaller_set_size > 0 else 0

def calculate_dice_coefficient(arr1, arr2):
    intersection = np.minimum(arr1, arr2).sum()
    total = arr1.sum() + arr2.sum()
    return 2 * intersection / total if total > 0 else 0

def calculate_metrics(train_qgrams, test_qgrams, pred_qgrams, incorrect_pred_qgrams):
    qgrams_union = set(train_qgrams) | set(test_qgrams) | set(pred_qgrams) | set(incorrect_pred_qgrams)
    vectors = [np.array([qgrams.get(qgram, 0) for qgram in qgrams_union]) for qgrams in (train_qgrams, test_qgrams, pred_qgrams, incorrect_pred_qgrams)]
    
    metrics = {}
    for name, func in [("Spearman Correlation", spearmanr), ("Frequency Similarity", calculate_frequency_similarity),
                       ("Dice Coefficient", calculate_dice_coefficient), ("Intersection Count", calculate_intersection_count)]:
        for i, set1 in enumerate(["Train", "Train", "Train"]):
            for j, set2 in enumerate(["Correct Pred", "Test", "Incorrect Pred"]):
                if i < j:
                    metric_name = f"{name} {set1}-{set2}"
                    result = func(vectors[0], vectors[j])[0] if func == spearmanr else func(vectors[0], vectors[j])
                    metrics[metric_name] = 0 if np.isnan(result) else result
    
    return metrics

def ensure_nltk_corpus(corpus_name):
    try:
        nltk.data.find(f'corpora/{corpus_name}')
    except LookupError:
        nltk.download(corpus_name)

def process_corpus(corpus_name, config):
    try:
        print(f"Processing {corpus_name} with q_range: {config.q_range}")
        corpus_file = config.corpus_dir / f'{corpus_name}.txt'
        if not corpus_file.exists():
            if corpus_name in ['brown', 'cmudict']:
                ensure_nltk_corpus(corpus_name)
                with open(corpus_file, 'w', encoding='utf-8') as f:
                    f.write(' '.join(nltk.corpus.brown.words()) if corpus_name == 'brown' else '\n'.join(nltk.corpus.cmudict.words()))
            else:
                raise FileNotFoundError(f"Corpus file for {corpus_name} not found and it's not an NLTK corpus.")

        corpus_manager = CorpusManager(corpus_name, config)
        eval_model = EvaluateModel(corpus_manager)
        
        process_and_save_qgrams(corpus_manager.train_set, config.get_file_path(corpus_name, 'qgram', 'train'), config)
        process_and_save_qgrams({word for _, _, word in corpus_manager.test_set}, config.get_file_path(corpus_name, 'qgram', 'test'), config)
        
        predictions = eval_model.evaluate_character_predictions(eval_model.prediction_method)[1]
        correct_words = []
        incorrect_words = []
        for missing_letter, correct_letter, orig_word, preds, _ in predictions:
            if preds[0][0] == correct_letter:
                correct_words.append(orig_word)
            else:
                incorrect_words.append(orig_word)
        
        correct_qgrams = Counter(qgram for word in correct_words for qgram in generate_qgrams(word, config.q_range))
        incorrect_qgrams = Counter(qgram for word in incorrect_words for qgram in generate_qgrams(word, config.q_range))
        
        for qgrams, suffix in [(correct_qgrams, "correct"), (incorrect_qgrams, "incorrect")]:
            write_qgram_frequencies_to_file(qgrams, config.qgrams_dir / f"{corpus_name}_{suffix}_qgrams.txt")
        
        train_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'train'))
        test_qgrams = read_qgram_frequencies(config.get_file_path(corpus_name, 'qgram', 'test'))
        
        metrics = calculate_metrics(train_qgrams, test_qgrams, correct_qgrams, incorrect_qgrams)
        log_results(corpus_name, metrics)

    except Exception as e:
        logging.error(f"Error processing {corpus_name}: {e}")
        raise  # Re-raise the exception to see the full traceback

def log_results(corpus_name, metrics):
    separator = '-' * 50
    header = f"{corpus_name} Corpus Analysis"
    logging.info(f'\n{separator}\n{header}\n{separator}')

    last_category = None
    for metric_name in sorted(metrics.keys()):
        current_category = metric_name.split(" ")[0]
        if last_category is not None and current_category != last_category:
            logging.info("")
        logging.info(f"{metric_name}: {metrics[metric_name]:.4f}")
        last_category = current_category

def main():
    config = Config(Path.cwd())
    setup_logging(config)
    os.makedirs(config.corpus_dir, exist_ok=True)
    for corpus in config.corpora:
        process_corpus(corpus, config)

if __name__ == "__main__":
    main()