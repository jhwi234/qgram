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
from nltk.corpus import brown, gutenberg, reuters
from nltk.tokenize import word_tokenize, RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import linregress
import statsmodels.api as sm

class LoggerConfig:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

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
    def __init__(self, corpus_source, allow_download=True, custom_download_dir=None):
        self.corpus_source = corpus_source
        self.allow_download = allow_download
        self.custom_download_dir = custom_download_dir
        self.download_attempted = False  # Flag to track download attempts

    def is_corpus_available(self):
        try:
            nltk.data.find(self.corpus_source)
            return True
        except LookupError:
            return False

    def download_corpus(self):
        if self.custom_download_dir:
            nltk.data.path.append(self.custom_download_dir)
        nltk.download(self.corpus_source, download_dir=self.custom_download_dir, quiet=True)
        self.download_attempted = True

    def load_corpus(self):
        """
        Load the entire corpus into a list of tokens.
        """
        if os.path.isfile(self.corpus_source) or os.path.isdir(self.corpus_source):
            # Handle local file or directory
            corpus_reader = PlaintextCorpusReader(self.corpus_source, '.*')
            return [token.lower() for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]
        elif self.is_corpus_available():
            # Load from NLTK if available
            corpus_reader = getattr(nltk.corpus, self.corpus_source)
            return [token.lower() for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]
        elif self.allow_download and not self.download_attempted:
            # Attempt to download if not already attempted
            self.download_corpus()
            corpus_reader = getattr(nltk.corpus, self.corpus_source)
            return [token.lower() for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]
        else:
            # Corpus not found and either download is disabled or already attempted
            raise RuntimeError(f"Failed to access or download the NLTK corpus: {self.corpus_source}")
class Tokenizer:
    def __init__(self, remove_stopwords=False, remove_punctuation=False, use_nltk_tokenizer=False, stopwords_language='english'):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.stopwords_language = stopwords_language
        self.custom_regex = None
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        if self.remove_stopwords:
            try:
                nltk.data.find(f'corpora/stopwords/{self.stopwords_language}')
            except LookupError:
                nltk.download('stopwords', quiet=True)

    def set_custom_regex(self, pattern):
        """Set a custom regex pattern for tokenization."""
        try:
            self.custom_regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def _remove_stopwords_and_punctuation(self, tokens):
        if not self.remove_stopwords and not self.remove_punctuation:
            return tokens

        stop_words = set(stopwords.words(self.stopwords_language)) if self.remove_stopwords else set()
        punctuation = set(string.punctuation) if self.remove_punctuation else set()

        return [token for token in tokens if token not in punctuation and token not in stop_words]

    def tokenize(self, text) -> list:
        # If text is a list, join it into a single string
        if isinstance(text, list):
            text = ' '.join(text)

        # Perform initial tokenization
        if self.custom_regex:
            tokens = self.custom_regex.findall(text)
        elif self.use_nltk_tokenizer:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()

        # Apply removal of stopwords and punctuation
        return self._remove_stopwords_and_punctuation(tokens)
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
            # Check if rank is within the valid range
            if 1 <= query <= len(self.sorted_tokens):
                token, freq = self.sorted_tokens[query - 1]
                return token, freq, query
            else:
                raise ValueError(f"Rank {query} is out of range (1 to {len(self.sorted_tokens)}).")
        elif isinstance(query, str):
            word = query.lower()
            freq = self.frequency.get(word, 0)
            rank = self.token_ranks.get(word, None)
            return word, freq, rank
        else:
            raise TypeError("Query must be a word (str) or a rank (int).")

    def get_words_in_rank_range(self, start_rank, end_rank):
        # Validate the rank range
        if not (1 <= start_rank <= len(self.sorted_tokens)):
            raise ValueError(f"Start rank {start_rank} is out of range (1 to {len(self.sorted_tokens)}).")
        if not (start_rank <= end_rank <= len(self.sorted_tokens)):
            raise ValueError(f"End rank {end_rank} is out of the valid range ({start_rank} to {len(self.sorted_tokens)}).")

        # Extract tokens within the specified rank range
        return [(token, freq, rank) for rank, (token, freq) in enumerate(self.sorted_tokens, start=1) if start_rank <= rank <= end_rank]

class AdvancedCorpusAnalyzer(BasicCorpusAnalyzer):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.cum_freqs, self.token_ranks = self._calculate_cumulative_frequencies()

    def cumulative_frequency_analysis(self, lower_percent=0, upper_percent=100) -> list:
        """
        Get words, their frequencies, and ranks that fall within a specified frequency range.
        Validates input percentages to ensure they are within the 0-100 range and lower_percent is not greater than upper_percent.
        """
        # Validate input percentages
        if not 0 <= lower_percent <= 100:
            raise ValueError("lower_percent must be between 0 and 100.")
        if not 0 <= upper_percent <= 100:
            raise ValueError("upper_percent must be between 0 and 100.")
        if lower_percent > upper_percent:
            raise ValueError("lower_percent must not be greater than upper_percent.")

        if not self.tokens:
            return []

        total = sum(self.frequency.values())
        lower_threshold = total * (lower_percent / 100)
        upper_threshold = total * (upper_percent / 100)

        words_in_range = []
        for token, freq in self.sorted_tokens:
            cumulative_freq = self.cum_freqs[token]
            if lower_threshold <= cumulative_freq <= upper_threshold:
                words_in_range.append((token, freq, self.token_ranks[token], cumulative_freq))
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
        self.sorted_tokens = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        self.total_token_count = sum(self.frequency.values())

    @lru_cache(maxsize=None)
    def _calculate_generalized_harmonic(self, n, alpha):
        return sum(1 / (i ** alpha) for i in range(1, n + 1))

    def plot_comparison(self, alpha=1):
        ranks = np.arange(1, len(self.sorted_tokens) + 1)
        frequencies = np.array([freq for _, freq in self.sorted_tokens])
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        harmonic_number = self._calculate_generalized_harmonic(len(ranks), alpha)
        ideal_zipfian = (self.total_token_count / harmonic_number) / np.power(ranks, alpha)
        log_ideal_zipfian = np.log(ideal_zipfian)
        plt.figure(figsize=(10, 6))
        plt.scatter(log_ranks, log_freqs, color='blue', label='Actual Frequencies', s=10)
        plt.plot(log_ranks, log_ideal_zipfian, 'g--', label='Ideal Zipfian', linewidth=2)
        plt.xlabel('Log of Rank')
        plt.ylabel('Log of Frequency')
        plt.title('Comparison of Actual Frequencies with Ideal Zipfian')
        plt.legend()
        plt.show()

    def assess_zipfian_fit(self, alpha=1):
        deviations = self._calculate_zipfian_deviations(alpha)
        mean_deviation = np.mean(np.abs(deviations))
        std_deviation = np.std(deviations)
        return mean_deviation, std_deviation

    def _calculate_zipfian_deviations(self, alpha=1):
        n = len(self.sorted_tokens)
        ranks = np.arange(1, n + 1)
        harmonic_number = self._calculate_generalized_harmonic(n, alpha)
        harmonic_factor = self.total_token_count / harmonic_number
        expected_freqs = harmonic_factor / np.power(ranks, alpha)
        actual_freqs = np.array([freq for _, freq in self.sorted_tokens])
        deviations = actual_freqs - expected_freqs
        return deviations

def analyze_corpus(corpus_name):
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing corpus: {corpus_name}")

    try:
        # Check the type of corpus and load it accordingly
        if corpus_name == 'brown':
            tokens = brown.words()
        elif corpus_name == 'gutenberg':
            tokens = gutenberg.words()
        elif corpus_name == 'reuters':
            tokens = reuters.words()
        else:
            raise ValueError(f"Corpus {corpus_name} is not recognized.")

        # Tokenize the corpus
        tokenizer = Tokenizer(remove_punctuation=True)
        tokens = tokenizer.tokenize(' '.join(tokens))

        # Instantiate AdvancedCorpusAnalyzer
        advanced_analyzer = AdvancedCorpusAnalyzer(tokens)

        # Perform advanced corpus analysis
        yules_k, yules_interpretation = advanced_analyzer.yules_k()
        logger.info(f"Yule's K Measure: {yules_k:.4f}, Interpretation: {yules_interpretation}")

        herdans_c, herdans_interpretation = advanced_analyzer.herdans_c()
        logger.info(f"Herdan's C Measure: {herdans_c:.4f}, Interpretation: {herdans_interpretation}")

        # Perform the Zipfian analysis
        zipfian_analyzer = ZipfianAnalysis(tokens)
        zipfian_analyzer.plot_comparison(alpha=1)
        mean_dev, std_dev = zipfian_analyzer.assess_zipfian_fit(alpha=1)
        logger.info(f"Mean deviation from Ideal Zipfian: {mean_dev:.4f}")
        logger.info(f"Standard deviation from Ideal Zipfian: {std_dev:.4f}")

    except Exception as e:
        logger.error(f"An error occurred while analyzing {corpus_name}: {e}")

def comprehensive_corpus_analysis():
    logger_config = LoggerConfig()
    logger_config.setup_logging()

    # List of nltk corpora
    corpora = ['brown', 'gutenberg', 'reuters']
    for corpus_name in corpora:
        analyze_corpus(corpus_name)

if __name__ == '__main__':
    comprehensive_corpus_analysis()
