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
from nltk.tokenize import word_tokenize, RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

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
    def __init__(self, remove_stopwords=False, remove_punctuation=False, use_nltk_tokenizer=False):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.custom_regex = None
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        if self.remove_stopwords:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

    def set_custom_regex(self, pattern):
        """Set a custom regex pattern for tokenization."""
        self.custom_regex = re.compile(pattern)

    def _remove_stopwords_and_punctuation(self, tokens):
        punctuation = set(string.punctuation)
        stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()
        return [token for token in tokens if token not in punctuation and token not in stop_words]

    def tokenize(self, text) -> list:
        # If text is a list, join it into a single string
        if isinstance(text, list):
            text = ' '.join(text)

        # Initial tokenization
        if self.custom_regex:
            tokens = self.custom_regex.findall(text)
        elif self.use_nltk_tokenizer:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()

        # Remove stopwords and punctuation
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
        # Calculate cumulative frequencies once during initialization
        self.cum_freqs, self.token_ranks = self._calculate_cumulative_frequencies()

    def cumulative_frequency_analysis(self, lower_percent=0, upper_percent=100) -> list:
        """
        Get words, their frequencies, and ranks that fall within a specified frequency range.
        """
        if not self.tokens:
            return []  # Handle empty token list

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
class ZipfianAnalysis(BasicCorpusAnalyzer):
    def __init__(self, tokens):
        super().__init__(tokens)
        # Ensure that the tokens list is not empty for analysis
        if not tokens:
            raise ValueError("Token list is empty. Zipfian analysis requires a non-empty list of tokens.")

    @lru_cache(maxsize=None)
    def _calculate_generalized_harmonic(self, n, alpha) -> float:
        """
        Calculate the generalized harmonic number of order 'n' for 'alpha'.
        This is used in calculating expected frequencies based on Zipf's Law.
        """
        # Summation of 1/(i^alpha) for i from 1 to n
        return sum(1 / (i ** alpha) for i in range(1, n + 1))

    def _calculate_zipfian_deviations(self, alpha=1) -> np.ndarray:
        """
        Calculate the deviations of actual word frequencies from the expected Zipfian frequencies.
        Uses vectorized operations for efficiency.
        """
        n = len(self.sorted_tokens)
        if n == 0:
            raise ValueError("No sorted tokens available for analysis.")

        # Generate an array of ranks (1 to n)
        ranks = np.arange(1, n + 1)

        # Compute the generalized harmonic number
        harmonic_number = self._calculate_generalized_harmonic(n, alpha)

        # Calculate the harmonic factor for expected frequency computation
        harmonic_factor = self.total_token_count / harmonic_number

        # Vectorized computation of expected frequencies based on Zipf's Law
        expected_freqs = harmonic_factor / np.power(ranks, alpha)

        # Extract actual frequencies and compute deviations from expected frequencies
        actual_freqs = np.array([freq for _, freq in self.sorted_tokens])
        deviations = actual_freqs - expected_freqs

        # Combine results into a structured array: ranks, actual frequencies, expected frequencies, deviations
        return np.column_stack((ranks, actual_freqs, expected_freqs, deviations))

    def compare_with_ideal_zipf(self, alpha=1) -> dict:
        """
        Compare the actual frequency distribution of the corpus with the ideal Zipfian distribution.
        Includes a comparison of the regression line with the ideal Zipfian line.
        """
        # Perform regression analysis on actual frequencies
        regression_results = self.fit_zipfian_regression()

        # The ideal Zipfian slope for alpha=1 is -1
        ideal_slope = -alpha

        # Compare actual slope with ideal slope
        slope_deviation = regression_results['slope'] - ideal_slope

        return {
            "actual_regression": regression_results,
            "ideal_slope": ideal_slope,
            "slope_deviation": slope_deviation
        }

    def summarize_zipfian_compliance(self, alpha=1) -> dict:
        """
        Summarize how closely the corpus follows Zipf's Law.
        Focuses on the comparison of the regression line of actual frequencies with the ideal Zipfian line.
        """
        zipfian_data = self.compare_with_ideal_zipf(alpha)
        slope_deviation = zipfian_data['slope_deviation']
        regression_results = self.fit_zipfian_regression()
        # Interpret the results based on slope deviation and R-value
        interpretation = self._interpret_compliance(slope_deviation, regression_results['r_value'])

        return {
            "slope_deviation": slope_deviation,
            "r_value": regression_results['r_value'],
            "p_value": regression_results['p_value'],
            "std_err": regression_results['std_err'],
            "interpretation": interpretation,
            "actual_regression": regression_results  # Include this line
        }

    def _interpret_compliance(self, slope_deviation, r_value) -> str:
        """
        Provide a textual interpretation of the compliance based on slope deviation and R-value.
        """
        if abs(slope_deviation) < 0.1 and r_value > 0.9:
            return "Excellent adherence to Zipf's Law."
        elif abs(slope_deviation) < 0.2 and r_value > 0.8:
            return "Good adherence to Zipf's Law with some deviations."
        elif abs(slope_deviation) < 0.3 and r_value > 0.7:
            return "Moderate adherence to Zipf's Law, notable deviations observed."
        else:
            return "Low adherence to Zipf's Law, significant deviations from expected frequency distribution."

    @staticmethod
    def power_law(x, a, b):
        """
        Power-law function for curve fitting.
        """
        return a * np.power(x, b)

    def fit_power_law(self):
        """
        Fit a power-law distribution to the ranks and frequencies.
        """
        # Extract ranks and frequencies
        ranks = np.arange(1, len(self.sorted_tokens) + 1)
        frequencies = np.array([freq for _, freq in self.sorted_tokens])

        # Perform the curve fit
        params, _ = curve_fit(self.power_law, ranks, frequencies, p0=[1, -1])
        return params

    def plot_zipfian_comparison(self, alpha=1):
        """
        Plot the actual frequencies, fitted power-law, and ideal Zipfian distribution.
        """
        # Extract ranks and frequencies
        ranks = np.arange(1, len(self.sorted_tokens) + 1)
        frequencies = np.array([freq for _, freq in self.sorted_tokens])

        # Fit power-law distribution
        fitted_params = self.fit_power_law()

        # Calculate expected frequencies using the fitted power-law parameters
        fitted_freqs = self.power_law(ranks, *fitted_params)

        # Calculate ideal Zipfian frequencies using a harmonic number
        harmonic_number = self._calculate_generalized_harmonic(len(ranks), alpha)
        zipfian_freqs = (self.total_token_count / harmonic_number) / np.power(ranks, alpha)

        # Plot the frequencies
        plt.figure(figsize=(10, 6))
        plt.loglog(ranks, frequencies, 'bo', label='Actual Frequencies', markersize=5)
        plt.loglog(ranks, fitted_freqs, 'r-', label='Fitted Power-Law', linewidth=2)
        plt.loglog(ranks, zipfian_freqs, 'g--', label='Ideal Zipfian', linewidth=2)

        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title('Comparison of Actual, Fitted Power-Law, and Ideal Zipfian Frequencies')
        plt.legend()
        plt.show()

def main():
    logger_config = LoggerConfig()
    logger = logger_config.setup_logging()

    # Load and tokenize the corpus
    loader = CorpusLoader('brown')
    logger.info(f"CORPUS ANALYSIS REPORT FOR '{loader.corpus_source.upper()}'")
    tokenizer = Tokenizer(remove_punctuation=True, use_nltk_tokenizer=True)
    tokens = tokenizer.tokenize(loader.load_corpus())

    # Basic analysis of the corpus
    basic_analyzer = BasicCorpusAnalyzer(tokens)
    median_token, median_freq = basic_analyzer.find_median_token()
    mode_token, mode_freq = basic_analyzer.mode_token()

    logger.info(f"Most Frequent Token: '{mode_token}' (Frequency: {mode_freq})")
    logger.info(f"Median Token: '{median_token}' (Frequency: {median_freq})")
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
    logger.info("FITTING POWER-LAW DISTRIBUTION")
    fitted_params = zipfian_analyzer.fit_power_law()
    logger.info(f"Fitted Power-Law Parameters: a = {fitted_params[0]}, b = {fitted_params[1]}")

    # Plotting the comparison
    zipfian_analyzer.plot_zipfian_comparison(alpha=1)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()