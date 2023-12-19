# Standard library imports
import logging
import os
import string
import math
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path
import regex as re

# Third-party imports
import nltk
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import numpy as np

class LoggerConfig:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)  # Create the log directory if it doesn't exist

    def setup_logging(self):
        # Setup log file handler
        logfile = self.log_dir / 'corpus_analysis.log'
        file_handler = logging.FileHandler(logfile, mode='a')
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')
        file_handler.setFormatter(file_format)
        file_handler.setLevel(logging.INFO)

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        console_handler.setLevel(logging.INFO)

        # Configure logging with both handlers
        logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

        return logging.getLogger(__name__)
class CorpusLoader:
    def __init__(self, corpus_source):
        self.corpus_source = corpus_source

    def load_corpus(self):
        """
        Load the entire corpus into a list of tokens.
        """
        if os.path.isfile(self.corpus_source) or os.path.isdir(self.corpus_source):
            corpus_reader = PlaintextCorpusReader(self.corpus_source, '.*')
            return [token.lower() for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]
        else:
            try:
                nltk.data.find(self.corpus_source)
                corpus_reader = getattr(nltk.corpus, self.corpus_source)
                return [token.lower() for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]
            except LookupError:
                nltk.download(self.corpus_source, quiet=True)
                corpus_reader = getattr(nltk.corpus, self.corpus_source)
                return [token.lower() for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]
class Tokenizer:
    def __init__(self, remove_stopwords=False, remove_punctuation=False, custom_regex=None):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.custom_regex = re.compile(custom_regex) if custom_regex else None
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        if self.remove_stopwords:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

    def tokenize(self, tokens) -> list:
        if self.remove_stopwords or self.remove_punctuation:
            punctuation_set = set(string.punctuation) if self.remove_punctuation else set()
            stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()
            tokens = [token for token in tokens if token not in punctuation_set and token not in stop_words]

        if self.custom_regex:
            joined_text = ' '.join(tokens)
            tokens = self.custom_regex.findall(joined_text)

        return tokens
class BasicCorpusAnalyzer:
    def __init__(self, tokens):
        self.tokens = tokens
        self.frequency = Counter(tokens)
        self.sorted_tokens = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        self.total_token_count = sum(self.frequency.values())

    def _calculate_cumulative_frequencies(self):
        cumulative = 0
        cum_freqs = {}
        token_ranks = {}
        for rank, (token, freq) in enumerate(self.sorted_tokens, start=1):
            cumulative += freq
            cum_freqs[token] = cumulative
            token_ranks[token] = rank
        return cum_freqs, token_ranks

    def find_median_token(self):
        median_index = self.total_token_count / 2
        cumulative = 0
        for token, freq in self.sorted_tokens:
            cumulative += freq
            if cumulative >= median_index:
                return token, freq

    def mode_token(self):
        return self.sorted_tokens[0]

    def mean_token_frequency(self):
        return self.total_token_count / len(self.frequency)

    def type_token_ratio(self):
        return len(self.frequency) / self.total_token_count

    def query_by_word_or_rank(self, query):
        if isinstance(query, int):
            if 1 <= query <= len(self.sorted_tokens):
                return self.sorted_tokens[query - 1]
            raise ValueError(f"Rank {query} is out of range.")
        elif isinstance(query, str):
            word = query.lower()
            return word, self.frequency.get(word, 0)
        raise TypeError("Query must be a word (str) or a rank (int).")

    def get_words_in_rank_range(self, start_rank, end_rank):
        if 1 <= start_rank <= end_rank <= len(self.sorted_tokens):
            return self.sorted_tokens[start_rank - 1:end_rank]
        raise ValueError("Invalid rank range.")
class AdvancedCorpusAnalyzer(BasicCorpusAnalyzer):
    def __init__(self, tokens):
        super().__init__(tokens)

    def cumulative_frequency_analysis(self, lower_percent=0, upper_percent=100) -> list:
        """
        Get words, their frequencies, and ranks that fall within a specified frequency range.
        """
        if not self.tokens:
            return []  # Handle empty token list

        cum_freqs, token_ranks = self.cumulative_frequencies, self.token_ranks

        total = sum(self.frequency.values())
        lower_threshold = total * (lower_percent / 100)
        upper_threshold = total * (upper_percent / 100)

        cumulative_freq = 0
        words_in_range = []
        for token, freq in self.sorted_tokens:
            cumulative_freq += freq
            if lower_threshold <= cumulative_freq <= upper_threshold:
                words_in_range.append((token, freq, token_ranks[token], cumulative_freq))
            elif cumulative_freq > upper_threshold:
                break

        return words_in_range

    def yules_k(self) -> tuple:
        # M1: Sum of the frequencies of all words in the corpus.
        # It represents the total number of word occurrences.
        M1 = sum(self.frequency.values())

        # M2: Sum of the squares of the frequencies of all words.
        # This emphasizes the contribution of high-frequency words.
        M2 = sum(f ** 2 for f in self.frequency.values())

        # K: Yule's K measure, calculated based on M1 and M2.
        # A lower K value indicates higher lexical diversity.
        K = 10**4 * (M2 - M1) / (M1 ** 2)

        # Standard deviation of word frequencies is used to scale the interpretation thresholds.
        # This accounts for the distribution of word usage in the corpus.
        std_dev = statistics.stdev(self.frequency.values())

        # Scaled thresholds for interpretation.
        # A high std_dev (indicating uneven word distribution) lowers the thresholds,
        # acknowledging that higher Yule's K values might still indicate reasonable diversity.
        scaled_low = 100 / std_dev
        scaled_high = 200 / std_dev

        # Interpretation based on scaled thresholds.
        # The interpretation dynamically adjusts to the corpus's distribution of word usage.
        if K < scaled_low:
            interpretation = "high lexical diversity"
        elif scaled_low <= K <= scaled_high:
            interpretation = "moderate lexical diversity"
        else:
            interpretation = "low lexical diversity"

        return K, interpretation

    def herdans_c(self) -> tuple:
        # N: Total number of words in the corpus.
        # V: Number of unique words (vocabulary size).
        N = sum(self.frequency.values())  # Total number of words
        V = len(self.frequency)  # Vocabulary size

        # C: Herdan's C value, a logarithmic measure of vocabulary richness.
        # Higher C values indicate richer vocabulary.
        C = math.log(V) / math.log(N)

        # Dynamic scaling factor is calculated using median and mean word counts.
        # This makes the scaling factor sensitive to the typical word distribution in the corpus.
        median_word_count = statistics.median(self.frequency.values())
        mean_word_count = statistics.mean(self.frequency.values())

        # The dynamic factor is calculated as the logarithm of the sum of the median and mean word counts,
        # normalized by log(2). This normalization makes the factor adjust appropriately across
        # different corpus sizes and distributions.
        dynamic_factor = math.log(median_word_count + mean_word_count) / math.log(2)

        # Scaled thresholds for interpretation.
        # The thresholds adjust based on the dynamic factor, allowing for flexible interpretation
        # across different corpus sizes and types.
        scaled_very_rich = 0.8 / dynamic_factor
        scaled_rich = 0.6 / dynamic_factor

        # Interpretation based on scaled thresholds.
        # The interpretation dynamically adjusts to the size and distribution of the corpus.
        if C > scaled_very_rich:
            interpretation = "very rich vocabulary"
        elif scaled_rich <= C <= scaled_very_rich:
            interpretation = "rich vocabulary"
        else:
            interpretation = "limited vocabulary richness"

        return C, interpretation
class ZipfianAnalysis:
    def __init__(self, tokens):
        self.tokens = tokens
        self.frequency = Counter(tokens)
        # Ensure sorting is in descending order by frequency
        self.sorted_tokens = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        self.total_token_count = sum(self.frequency.values())

    def _calculate_generalized_harmonic(self, n, alpha) -> float:
        return sum(1 / (i ** alpha) for i in range(1, n + 1))

    def _calculate_zipfian_deviations(self, alpha=1) -> list:
        n = len(self.sorted_tokens)
        harmonic_number = self._calculate_generalized_harmonic(n, alpha)
        harmonic_factor = self.total_token_count / harmonic_number

        zipfian_deviations = []
        for rank, (_, actual_freq) in enumerate(self.sorted_tokens, 1):
            expected_freq = harmonic_factor / (rank ** alpha)
            deviation = actual_freq - expected_freq
            zipfian_deviations.append((rank, actual_freq, expected_freq, deviation))

        return zipfian_deviations

    def zipfs_law_analysis(self, alpha=1, frequency_range=None) -> tuple:
        """
        Analyze the corpus based on Zipf's Law with optional frequency range limit.
        """
        zipfian_deviations = self._calculate_zipfian_deviations(alpha)

        if frequency_range:
            zipfian_deviations = [dev for dev in zipfian_deviations if frequency_range[0] <= dev[0] <= frequency_range[1]]

        deviations = [dev[3] for dev in zipfian_deviations]
        mean_deviation = sum(deviations) / len(deviations)
        median_deviation = statistics.median(deviations)
        std_deviation = statistics.stdev(deviations)

        return mean_deviation, median_deviation, std_deviation

    def report_top_zipfian_deviations(self, top_n=10) -> dict:
        """
        Report the words with the highest positive and negative deviations from Zipf's Law.
        """
        zipfian_deviations = self._calculate_zipfian_deviations()
        sorted_deviations = sorted(zipfian_deviations, key=lambda x: x[3], reverse=True)

        top_positive = sorted_deviations[:top_n]
        top_negative = sorted_deviations[-top_n:]

        return {"top_positive": top_positive, "top_negative": top_negative}

    def categorize_zipfian_deviations(self, threshold=10) -> dict:
        """
        Categorize words based on their deviation from Zipf's Law.
        """
        zipfian_deviations = self._calculate_zipfian_deviations()
        categories = {'significantly_above': [], 'significantly_below': [], 'close_to_expected': []}

        for rank, actual_freq, expected_freq, deviation in zipfian_deviations:
            if deviation > threshold:
                categories['significantly_above'].append((rank, actual_freq, expected_freq, deviation))
            elif deviation < -threshold:
                categories['significantly_below'].append((rank, actual_freq, expected_freq, deviation))
            else:
                categories['close_to_expected'].append((rank, actual_freq, expected_freq, deviation))

        return categories

    def compare_with_ideal_zipf(self, alpha=1) -> dict:
        """
        Compare the actual frequency distribution with an ideal Zipfian distribution.
        """
        harmonic_number = self._calculate_generalized_harmonic(len(self.sorted_tokens), alpha)

        zipfian_comparison = {'rank': [], 'actual_frequency': [], 'expected_zipf_frequency': [], 'deviation': []}
        for rank, (_, actual_freq) in enumerate(self.sorted_tokens, 1):
            expected_freq = self.total_tokens / (rank ** alpha * harmonic_number)
            deviation = actual_freq - expected_freq

            zipfian_comparison['rank'].append(rank)
            zipfian_comparison['actual_frequency'].append(actual_freq)
            zipfian_comparison['expected_zipf_frequency'].append(expected_freq)
            zipfian_comparison['deviation'].append(deviation)

        return zipfian_comparison

def main():
    logger_config = LoggerConfig()
    logger = logger_config.setup_logging()

    # Load and tokenize the corpus
    loader = CorpusLoader('brown')
    logger.info(f"CORPUS ANALYSIS REPORT FOR '{loader.corpus_source.upper()}'")
    tokenizer = Tokenizer(remove_punctuation=True)
    # After tokenization
    tokens = tokenizer.tokenize(loader.load_corpus())
    # Basic analysis of the corpus
    basic_analyzer = BasicCorpusAnalyzer(tokens)
    median_token, median_freq = basic_analyzer.find_median_token()
    mode_token, mode_freq = basic_analyzer.mode_token()

    logger.info(f"Median Token: '{median_token}' (Frequency: {median_freq})")
    logger.info(f"Most Frequent Token: '{mode_token}' (Frequency: {mode_freq})")
    logger.info(f"Average Token Frequency: {basic_analyzer.mean_token_frequency():.2f} tokens")
    logger.info(f"Type-Token Ratio: {basic_analyzer.type_token_ratio():.4f}")

    # Advanced analysis of the corpus
    advanced_analyzer = AdvancedCorpusAnalyzer(tokens)
    herdans_c_value, herdans_c_interpretation = advanced_analyzer.herdans_c()
    yules_k_value, yules_k_interpretation = advanced_analyzer.yules_k()

    logger.info(f"Herdan's C: {herdans_c_value:.4f} ({herdans_c_interpretation})")
    logger.info(f"Yule's K: {yules_k_value:.2f} ({yules_k_interpretation})")

    # Zipf's Law Analysis
    zipfian_analyzer = ZipfianAnalysis(tokens)
    mean_deviation, median_deviation, std_deviation = zipfian_analyzer.zipfs_law_analysis()

    logger.info("ZIPF'S LAW ANALYSIS")
    logger.info(f"Mean Deviation: {mean_deviation:.4f}")
    logger.info(f"Median Deviation: {median_deviation:.4f}")
    logger.info(f"Standard Deviation: {std_deviation:.4f}")

    logger.info('=' * 60)

if __name__ == '__main__':
    main()