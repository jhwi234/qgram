# Standard library imports
import logging
import os
import string
import math
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path

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
        self._corpus_data = None  # Initialize without loading data

    def load_corpus(self) -> list:
        if self._corpus_data is None:  # Load data only if it's not already loaded
            self._corpus_data = self._load_data()
        return self._corpus_data

    def _load_data(self) -> list:
        # Check if the source is a file or directory and load accordingly
        if os.path.isfile(self.corpus_source) or os.path.isdir(self.corpus_source):
            corpus_reader = PlaintextCorpusReader(self.corpus_source, '.*')
            return [token.lower() for token in corpus_reader.words()]
        try:
            # For named NLTK corpora, check if it's available or download it
            nltk.data.find(self.corpus_source)
        except LookupError:
            nltk.download(self.corpus_source, quiet=True)
        # Load the corpus from NLTK and convert all words to lowercase
        return [word.lower() for word in getattr(nltk.corpus, self.corpus_source).words()]

class Tokenizer:
    def __init__(self, remove_stopwords=False, remove_punctuation=False, custom_regex=None):
        self.remove_stopwords = remove_stopwords  # Flag to remove stopwords
        self.remove_punctuation = remove_punctuation  # Flag to remove punctuation
        self.custom_regex = custom_regex  # Optional custom regular expression for tokenization
        self._ensure_nltk_resources()  # Ensure necessary NLTK resources are downloaded

    def _ensure_nltk_resources(self):
        # Download NLTK stopwords if necessary
        if self.remove_stopwords and not nltk.data.find('corpora/stopwords'):
            nltk.download('stopwords')

    def tokenize(self, tokens) -> list:
        # Tokenize the text, optionally removing stopwords and punctuation
        if self.remove_stopwords or self.remove_punctuation:
            punctuation_set = set(string.punctuation)
            stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()
            tokens = [token for token in tokens if token not in punctuation_set and token not in stop_words]

        # Apply custom regex tokenizer if provided
        if self.custom_regex:
            tokenizer = RegexpTokenizer(self.custom_regex)
            tokens = tokenizer.tokenize(' '.join(tokens))

        return tokens

class Display:
    @staticmethod
    def display_token_info(description, token, freq, additional_info=None):
        info = f"{description}: '{token}' (Frequency: {freq})"
        if additional_info is not None:
            info += f", {additional_info}"
        print(info)

    @staticmethod
    def display_query_result(result):
        if isinstance(result, tuple) and len(result) == 2:
            print(f"{'Rank' if isinstance(result[0], int) else 'Frequency of'} '{result[0]}': {result[1]}")

    @staticmethod
    def display_words_in_rank_range(words_in_range, start_rank, end_rank):
        print(f"Words and Frequencies for Ranks {start_rank} to {end_rank}:")
        for rank, (word, freq) in enumerate(words_in_range, start=start_rank):
            print(f"  Rank {rank}: '{word}' (Frequency: {freq})")

    @staticmethod
    def display_token_frequency_distribution(frequency_distribution):
        if not frequency_distribution:
            print("No frequency distribution to display.")
            return

        # Prepare data for histogram
        freqs, counts = zip(*frequency_distribution)
        max_freq = max(freqs)

        plt.figure(figsize=(10, 6))

        # Create bins with logarithmic scale
        bins = np.logspace(np.log10(1), np.log10(max_freq), 50)
        plt.hist(freqs, bins=bins, weights=counts, log=True, color='skyblue', edgecolor='black')

        plt.xlabel('Token Frequencies (Log scale)')
        plt.ylabel('Number of Tokens (Log scale)')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.title('Token Frequency Distribution')
        plt.grid(True)
        plt.show()

    @staticmethod
    def display_dispersion_plot(tokens, words):
        if not tokens:
            print("The corpus is empty.")
            return

        text = nltk.Text(tokens)
        text.dispersion_plot(words)

class CorpusAnalyzer:
    def __init__(self, tokens):
        self.tokens = tokens
        self.frequency = Counter(tokens)
        self.sorted_tokens = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        self.total_token_count = sum(self.frequency.values())
        self.cumulative_frequencies, self.token_ranks = self._calculate_cumulative_frequencies()

    def _calculate_cumulative_frequencies(self) -> tuple:
        # Calculate cumulative frequencies and ranks for each token
        cumulative = 0
        cum_freqs = {}
        token_ranks = {}
        for rank, (token, freq) in enumerate(self.sorted_tokens, start=1):
            cumulative += freq
            cum_freqs[token] = cumulative
            token_ranks[token] = rank
        return cum_freqs, token_ranks

    def cumulative_frequency_analysis(self, percent_list) -> dict:
        # Get the words that represent certain percentage thresholds in terms of frequency
        if not self.tokens:
            return {}  # Handle empty token list

        results = {}
        total = sum(self.frequency.values())
        cumulative = 0
        sorted_percentages = sorted(percent_list)
        percent_index = 0
        threshold = total * (sorted_percentages[percent_index] / 100)

        for token, freq in self.sorted_tokens:
            cumulative += freq
            while cumulative >= threshold:
                results[sorted_percentages[percent_index]] = (token, freq)
                percent_index += 1
                if percent_index >= len(sorted_percentages):
                    return results
                threshold = total * (sorted_percentages[percent_index] / 100)

        return results

    @lru_cache(maxsize=128)  # Cache results for repeated queries
    def find_median_token(self) -> tuple:
        if not self.tokens:
            return None, 0, 0  # Handle empty token list
        median_index = self.total_token_count / 2
        for token, cum_freq in self.cumulative_frequencies.items():
            if cum_freq >= median_index:
                return token, self.frequency[token], self.token_ranks[token]

    def mode_token(self) -> tuple:
        # Get the most frequently occurring token in the corpus
        mode, freq = self.sorted_tokens[0]
        return mode, freq, 1

    def mean_token_frequency(self) -> float:
        # Calculate the average occurrence frequency of all tokens
        return self.total_token_count / len(self.frequency)

    def token_frequency_distribution(self) -> list:
        # Get the distribution of token frequencies
        frequency_count = Counter(self.frequency.values())
        return sorted(frequency_count.items())

    @lru_cache(maxsize=128)  # Cache results for repeated queries
    def type_token_ratio(self) -> float:
        # Calculate Type-Token Ratio (TTR)
        return len(self.frequency) / sum(self.frequency.values())

    def zipfs_law_analysis(self) -> list:
        # Analyze the corpus based on Zipf's Law
        return [(rank, freq, freq * rank) for rank, (_, freq) in enumerate(self.sorted_tokens, 1)]

    def hapax_legomena(self) -> list:
        # Get words that occur only once in the corpus
        return [word for word, freq in self.frequency.items() if freq == 1]

    @lru_cache(maxsize=128)  # Cache results for repeated queries
    def query_by_word_or_rank(self, query) -> tuple:
        if isinstance(query, int):
            if 1 <= query <= len(self.sorted_tokens):
                return self.sorted_tokens[query - 1]
            else:
                raise ValueError(f"Rank {query} is out of range.")
        elif isinstance(query, str):
            word = query.lower()
            if word in self.frequency:
                return word, self.frequency[word]
            else:
                raise ValueError(f"Word '{word}' not found in corpus.")
        else:
            raise TypeError("Query must be a word (str) or a rank (int).")

    def get_words_in_rank_range(self, start_rank, end_rank) -> list:
        # Get words within a specified rank range
        if 1 <= start_rank <= end_rank <= len(self.sorted_tokens):
            return self.sorted_tokens[start_rank - 1:end_rank]
        else:
            raise ValueError("Invalid rank range.")

    @lru_cache(maxsize=128)  # Cache results for repeated queries
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

    @lru_cache(maxsize=128)  # Cache results for repeated queries
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

def main():
    logger_config = LoggerConfig()
    logger = logger_config.setup_logging()

    # Load and tokenize the corpus
    loader = CorpusLoader('brown')
    logger.info(f"CORPUS ANALYSIS REPORT FOR {loader.corpus_source.upper()}")
    logger.info('=' * 40)
    tokenizer = Tokenizer(remove_stopwords=False, remove_punctuation=True)
    tokens = tokenizer.tokenize(loader.load_corpus())

    # Analyze the corpus
    analyzer = CorpusAnalyzer(tokens)

    median_token, freq, rank = analyzer.find_median_token()
    logger.info(f"\nMedian Token\n'{median_token}' | Frequency: {freq} | Rank: {rank}")
    mode_token, mode_freq, mode_rank = analyzer.mode_token()
    logger.info(f"\nMost Frequent Token\n'{mode_token}' | Frequency: {mode_freq} | Rank: {mode_rank}")
    logger.info(f"\nAverage Token Frequency: {analyzer.mean_token_frequency():.2f} tokens")
    logger.info(f"\nType-Token Ratio: {analyzer.type_token_ratio():.4f}")

    # Lexical Diversity
    herdans_c_value, herdans_c_interpretation = analyzer.herdans_c()
    logger.info(f"\nHerdan's C: {herdans_c_value:.4f} ({herdans_c_interpretation})")
    yules_k_value, yules_k_interpretation = analyzer.yules_k()
    logger.info(f"\nYule's K: {yules_k_value:.2f} ({yules_k_interpretation})")

    # Cumulative Frequency Analysis
    logger.info("\nCUMULATIVE FREQUENCY ANALYSIS\n")
    for percent, (word, freq) in analyzer.cumulative_frequency_analysis([50, 75, 90]).items():
        logger.info(f"Top {percent}% Word: '{word}' | Frequency: {freq}")

    # Zipf's Law Analysis
    logger.info("\nZIPF'S LAW ANALYSIS (Sample)\n")
    logger.info(f"{'Rank':<10} {'Frequency':<15} {'Product':<15}")
    for rank, freq, product in analyzer.zipfs_law_analysis()[:10]:
        logger.info(f"{rank:<10} {freq:<15} {product:<15}")

    # Hapax Legomena
    logger.info("\nHAPAX LEGOMENA (Sample)\n")
    for hapax in analyzer.hapax_legomena()[:10]:
        logger.info(f"- {hapax}")
    logger.info('=' * 40)

    # Visualization
    frequency_distribution = analyzer.token_frequency_distribution()
    Display.display_token_frequency_distribution(frequency_distribution)
    Display.display_dispersion_plot(tokens, ['the', 'man'])

if __name__ == "__main__":
    main()