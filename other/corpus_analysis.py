# Standard library imports
import os
import string
import math
import statistics
from collections import Counter
from functools import lru_cache
import regex as re

# Third-party imports
import nltk
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
class CorpusLoader:
    """
    Load a corpus from NLTK or a local file/directory.
    """
    def __init__(self, corpus_source, allow_download=True, custom_download_dir=None):
        self.corpus_source = corpus_source
        self.allow_download = allow_download
        self.custom_download_dir = custom_download_dir
        self.download_attempted = False  # Flag to track download attempts

    def is_corpus_available(self):
        """Check if the corpus is available locally or through NLTK."""
        try:
            nltk.data.find(self.corpus_source)
            return True
        except LookupError:
            return False

    def download_corpus(self):
        """Download the corpus from NLTK."""
        if self.custom_download_dir:
            nltk.data.path.append(self.custom_download_dir)
        nltk.download(self.corpus_source, download_dir=self.custom_download_dir, quiet=True)
        self.download_attempted = True

    def load_corpus(self) -> list:
        """Load the entire corpus into a list of tokens."""
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
    """
    Tokenize text into individual words.
    """
    def __init__(self, remove_stopwords=False, remove_punctuation=False, use_nltk_tokenizer=False, stopwords_language='english'):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.stopwords_language = stopwords_language
        self.custom_regex = None
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """Ensure that NLTK resources are available."""
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

    def _remove_stopwords_and_punctuation(self, tokens) -> list:
        """Remove stopwords and punctuation from a list of tokens."""
        if not self.remove_stopwords and not self.remove_punctuation:
            return tokens

        stop_words = set(stopwords.words(self.stopwords_language)) if self.remove_stopwords else set()
        punctuation = set(string.punctuation) if self.remove_punctuation else set()

        return [token for token in tokens if token not in punctuation and token not in stop_words]

    def tokenize(self, text) -> list:
        """Tokenize text into individual words."""
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
    """
    Analyze a corpus of text.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.frequency = Counter(tokens)
        self.sorted_tokens = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        self.total_token_count = sum(self.frequency.values())

    def _calculate_cumulative_frequencies(self) -> tuple:
        """Calculate cumulative frequencies and token ranks."""
        cumulative = 0
        cum_freqs = {}
        token_ranks = {}
        for rank, (token, freq) in enumerate(self.sorted_tokens, start=1):
            cumulative += freq
            cum_freqs[token] = cumulative
            token_ranks[token] = rank
        return cum_freqs, token_ranks

    def find_median_token(self) -> tuple:
        """Find the median token in the corpus."""
        median_index = self.total_token_count / 2
        cumulative = 0
        for token, freq in self.sorted_tokens:
            cumulative += freq
            if cumulative >= median_index:
                return token, freq

    def mode_token(self) -> tuple:
        """Find the mode token in the corpus."""
        return self.sorted_tokens[0]

    def mean_token_frequency(self) -> float:
        """Calculate the mean token frequency."""
        return self.total_token_count / len(self.frequency)

    def type_token_ratio(self) -> float:
        """Calculate the type-token ratio."""
        return len(self.frequency) / self.total_token_count

    def query_by_word_or_rank(self, query) -> tuple:
        """Query the corpus by word or rank."""
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

    def get_words_in_rank_range(self, start_rank, end_rank) -> list:
        """
        Get words, their frequencies, and ranks that fall within a specified rank range.
        Validates input ranks to ensure they are within the 1 to total number of words range,
        and that start_rank is not greater than end_rank.
        """
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
        self.N = sum(self.frequency.values())  # Store total word count for Herdan's C metric

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

    def yules_k(self) -> float:
        """
        Calculate Yule's K measure for lexical diversity.
        """
        # Convert frequency values to a NumPy array
        freqs = np.array(list(self.frequency.values()))

        # Utilize NumPy's vectorized operations for sum calculations
        M1 = np.sum(freqs)
        M2 = np.sum(freqs ** 2)

        # Calculate K using M1 and M2
        K = 10**4 * (M2 - M1) / (M1 ** 2)

        return K

    def herdans_c(self) -> float:
        """
        Calculate Herdan's C measure for vocabulary richness.
        """
        # V: Number of unique words (vocabulary size).
        V = len(self.frequency)

        # C: Herdan's C value, calculated using the stored total word count
        C = math.log(V) / math.log(self.N)

        return C
class ZipfianAnalysis:
    """
    Analyze a corpus to determine how well it fits the Zipfian distribution.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.frequency = Counter(tokens)
        self.sorted_tokens = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        self.total_token_count = sum(self.frequency.values())

    @lru_cache(maxsize=None)
    def _calculate_generalized_harmonic(self, n, alpha) -> float:
        """Calculate the generalized harmonic number."""
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

    def assess_zipfian_fit(self, alpha=1) -> tuple:
        """
        Assess how well the corpus fits the Zipfian distribution.
        Returns the mean and standard deviation of the absolute deviations
        between the actual and expected frequencies.
        """
        deviations = self._calculate_zipfian_deviations(alpha)
        mean_deviation = np.mean(np.abs(deviations))
        std_deviation = np.std(deviations)
        return mean_deviation, std_deviation

    def _calculate_zipfian_deviations(self, alpha=1) -> np.ndarray:
        """Calculate the deviations between actual and expected frequencies."""
        n = len(self.sorted_tokens)
        ranks = np.arange(1, n + 1)
        harmonic_number = self._calculate_generalized_harmonic(n, alpha)
        harmonic_factor = self.total_token_count / harmonic_number
        expected_freqs = harmonic_factor / np.power(ranks, alpha)
        actual_freqs = np.array([freq for _, freq in self.sorted_tokens])
        deviations = actual_freqs - expected_freqs
        return deviations
    
    def calculate_alpha(self) -> float:
        """
        Calculate the alpha parameter of the Zipfian distribution for the corpus.
        """
        ranks = np.arange(1, len(self.sorted_tokens) + 1)
        frequencies = np.array([freq for _, freq in self.sorted_tokens])

        # Applying log transformation to ranks and frequencies
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)

        # Perform linear regression on log-log data
        slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)

        # The slope of the line in log-log plot gives an estimate of -alpha
        alpha_estimate = -slope
        return alpha_estimate