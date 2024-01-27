# Standard library imports
import os
import string
import math
from collections import Counter
from functools import lru_cache
import regex as re

# Third-party imports
import nltk
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize, Bounds

class CorpusLoader:
    """
    Load a corpus from NLTK or a local file/directory, optimized for performance.
    """
    def __init__(self, corpus_source, allow_download=True, custom_download_dir=None):
        self.corpus_source = corpus_source
        self.allow_download = allow_download
        self.custom_download_dir = custom_download_dir
        self.corpus_cache = None

    def _download_corpus(self):
        """Download the corpus from NLTK."""
        if self.custom_download_dir:
            nltk.data.path.append(self.custom_download_dir)
        nltk.download(self.corpus_source, download_dir=self.custom_download_dir, quiet=True)

    def _load_corpus(self):
        """Load the corpus into memory."""
        if os.path.isfile(self.corpus_source) or os.path.isdir(self.corpus_source):
            corpus_reader = PlaintextCorpusReader(self.corpus_source, '.*')
        else:
            corpus_reader = getattr(nltk.corpus, self.corpus_source)
        return [token for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]

    def load_corpus(self):
        """Get the corpus, either from cache or by loading it."""
        if self.corpus_cache is not None:
            return self.corpus_cache

        if not self.is_corpus_available() and self.allow_download:
            self._download_corpus()

        self.corpus_cache = self._load_corpus()
        return self.corpus_cache

    def is_corpus_available(self):
        """Check if the corpus is available locally or through NLTK."""
        try:
            nltk.data.find(self.corpus_source)
            return True
        except LookupError:
            return False
class Tokenizer:
    """
    Tokenize text into individual words.
    """
    def __init__(self, remove_stopwords=False, remove_punctuation=False, use_nltk_tokenizer=True, stopwords_language='english'):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.stopwords_language = stopwords_language
        self.custom_regex = None
        self._unwanted_tokens = set()  # Combined set for stopwords and punctuation
        self._ensure_nltk_resources()
        self._load_unwanted_tokens()

    def _ensure_nltk_resources(self):
        """Ensure that NLTK resources are available."""
        if self.remove_stopwords:
            try:
                nltk.data.find(f'corpora/stopwords/{self.stopwords_language}')
            except LookupError:
                nltk.download('stopwords', quiet=True)

    def _load_unwanted_tokens(self):
        """Load stopwords and punctuation sets for efficient access."""
        if self.remove_stopwords:
            self._unwanted_tokens.update(stopwords.words(self.stopwords_language))
        if self.remove_punctuation:
            self._unwanted_tokens.update(string.punctuation)

    def set_custom_regex(self, pattern):
        """Set a custom regex pattern for tokenization with caching."""
        try:
            self.custom_regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def _remove_unwanted_tokens(self, tokens) -> list:
        """Remove unwanted tokens (stopwords, punctuation) from a list of tokens."""
        return [token for token in tokens if token not in self._unwanted_tokens and not token.startswith('``')]
    
    def tokenize(self, text, lowercase=False) -> list:
        """Tokenize text into individual words based on the selected method."""
        if isinstance(text, list):
            text = ' '.join(text)

        if lowercase:
            text = text.lower()

        # Perform tokenization based on the selected method
        if self.custom_regex:
            tokens = self.custom_regex.findall(text)
        elif self.use_nltk_tokenizer:
            tokens = word_tokenize(text)
        else:
            # Basic whitespace tokenization
            tokens = text.split()

        return self._remove_unwanted_tokens(tokens)
class BasicCorpusAnalyzer:
    """
    Analyze a corpus of text with optimized data structures.
    """
    def __init__(self, tokens, shuffle_tokens=False):
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("All tokens must be strings.")
        
        self.tokens = tokens
        if shuffle_tokens:
            self._shuffle_tokens()
            print("Tokens have been shuffled.")

        self.frequency = Counter(self.tokens)
        self._total_token_count = sum(self.frequency.values())
        self.token_details = self._initialize_token_details()

    def _shuffle_tokens(self):
        """Shuffle the tokens in the corpus using NumPy for efficiency."""
        np.random.shuffle(self.tokens)

    def _initialize_token_details(self):
        """
        Initialize the comprehensive dictionary with token details.
        """
        token_details = {}
        cumulative = 0
        for rank, (token, freq) in enumerate(self.frequency.most_common(), start=1):
            cumulative += freq
            token_details[token] = {
                'frequency': freq,
                'rank': rank,
                'cumulative_freq': cumulative
            }
        return token_details

    @property
    def total_token_count(self) -> int:
        """Return the total number of tokens in the corpus."""
        return self._total_token_count

    def find_median_token(self) -> dict:
        """Find the median token in the corpus."""
        median_index = self._total_token_count / 2
        cumulative = 0
        for token, details in self.token_details.items():
            cumulative += details['frequency']
            if cumulative >= median_index:
                return {'token': token, 'frequency': details['frequency']}

    def mean_token_frequency(self) -> float:
        """Calculate the mean token frequency."""
        return self.total_token_count / len(self.token_details)

    def query_by_token(self, token) -> dict:
        """Query the frequency and rank of a word in the corpus."""
        token = token.lower()
        details = self.token_details.get(token)
        if details is not None:
            return {'token': token, **details}
        else:
            raise ValueError(f"Token '{token}' not found in the corpus.")

    def query_by_rank(self, rank) -> dict:
        """Query the word and frequency of a rank in the corpus."""
        for token, details in self.token_details.items():
            if details['rank'] == rank:
                return {'token': token, 'frequency': details['frequency'], 'rank': rank}
        raise ValueError(f"Rank {rank} is out of range.")
    
    def cumulative_frequency_analysis(self, lower_percent=0, upper_percent=100):
        """Get the words in a certain cumulative frequency range."""
        if not 0 <= lower_percent <= 100 or not 0 <= upper_percent <= 100:
            raise ValueError("Percentages must be between 0 and 100.")

        # Swap the values if lower_percent is greater than upper_percent
        if lower_percent > upper_percent:
            lower_percent, upper_percent = upper_percent, lower_percent

        lower_threshold = self._total_token_count * (lower_percent / 100)
        upper_threshold = self._total_token_count * (upper_percent / 100)

        return [{'token': token, **details}
                for token, details in self.token_details.items()
                if lower_threshold <= details['cumulative_freq'] <= upper_threshold]

    def list_tokens_in_rank_range(self, start_rank, end_rank):
        """Get the words in a certain rank range."""
        if not (1 <= start_rank <= end_rank <= len(self.token_details)):
            raise ValueError("Rank range is out of valid bounds.")

        return [{'token': token, **details}
                for token, details in self.token_details.items()
                if start_rank <= details['rank'] <= end_rank]
class AdvancedCorpusAnalyzer(BasicCorpusAnalyzer):
    """
    Advanced analysis on a corpus of text, extending BasicCorpusAnalyzer.
    This class includes advanced linguistic metrics such as Yule's K measure,
    Herdan's C measure, and calculations related to Heaps' Law.
    """

    def __init__(self, tokens):
        super().__init__(tokens)

    def yules_k(self) -> float:
        """
        Calculate Yule's K measure for lexical diversity using NumPy.
        """
        freqs = np.array([details['frequency'] for details in self.token_details.values()])
        N = np.sum(freqs)
        sum_fi_fi_minus_1 = np.sum(freqs * (freqs - 1))
        K = 10**4 * (sum_fi_fi_minus_1 / (N * (N - 1))) if N > 1 else 0

        return K

    def herdans_c(self) -> float:
        """
        Calculate Herdan's C measure for vocabulary richness.
        Herdan's C is a measure of vocabulary richness in a text, representing
        the ratio of unique words to the total number of words.
        """
        V = len(self.token_details)
        N = self._total_token_count
        return math.log(V) / math.log(N)

    def calculate_heaps_law_sampling(self):
        """
        Calculate Heap's Law constants using a sampling approach with weighted linear regression.
        """
        if len(self.tokens) < 2:
            raise ValueError("Not enough data to calculate Heaps' law constants.")

        corpus_size = len(self.tokens)
        sampling_rate = 0.05
        sample_points = max(int(corpus_size * sampling_rate), 1000)

        sample_sizes = np.unique(np.logspace(0, np.log10(corpus_size), sample_points).astype(int))
        unique_word_counts = []
        unique_words = set()
        token_index = 0

        for size in sample_sizes:
            while token_index < size and token_index < corpus_size:
                unique_words.add(self.tokens[token_index])
                token_index += 1
            unique_word_counts.append(len(unique_words))

        # Pre-computing logarithms
        log_sample_sizes = np.log(sample_sizes)
        log_unique_word_counts = np.log(unique_word_counts)

        weights = np.sqrt(sample_sizes)
        beta, logK = np.polyfit(log_sample_sizes, log_unique_word_counts, 1, w=weights)
        K = np.exp(logK)

        return K, beta

    def calculate_heaps_law(self):
        """
        Calculate Heap's Law parameters using a standard approach.
        """
        if len(self.tokens) < 2:
            raise ValueError("Not enough tokens to calculate Heaps' law.")

        total_words = np.arange(1, len(self.tokens) + 1)
        unique_words = np.zeros(len(self.tokens))
        word_set = set()

        for i, token in enumerate(self.tokens):
            word_set.add(token)
            unique_words[i] = len(word_set)

        log_total_words = np.log(total_words)
        log_unique_words = np.log(unique_words)
        beta, logK = np.polyfit(log_total_words, log_unique_words, 1)
        K = np.exp(logK)

        return K, beta

    def plot_heaps_law(self, K_standard, beta_standard, K_sampling, beta_sampling, corpus_name):
        """
        Plot the results of Heap's Law calculations (both standard and sampling methods) for a given corpus.
        """
        total_words = np.arange(1, len(self.tokens) + 1)
        unique_words = []
        word_set = set()

        for token in self.tokens:
            word_set.add(token)
            unique_words.append(len(word_set))

        plt.figure(figsize=(10, 6))
        plt.plot(total_words, unique_words, label='Empirical Data', color='blue')
        plt.plot(total_words, K_standard * total_words ** beta_standard, '--', label=f'Standard Fit: K={K_standard:.2f}, beta={beta_standard:.2f}', color='green')
        plt.plot(total_words, K_sampling * total_words ** beta_sampling, '--', label=f'Sampling Fit: K={K_sampling:.2f}, beta={beta_sampling:.2f}', color='red')
        plt.xlabel('Total Words')
        plt.ylabel('Unique Words')
        plt.title(f"Heap's Law Analysis for {corpus_name} Corpus")
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_alpha(self, percentile_threshold=90) -> float:
        # Define the Zipf law function
        def zipf_func(rank, alpha, C):
            return C / rank**alpha
        
        # Extract frequencies and ranks from token details
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])
        ranks = np.array([details['rank'] for details in self.token_details.values()])
        
        # Determine threshold based on the specified percentile
        threshold = np.percentile(frequencies, percentile_threshold)
        high_freq_indices = frequencies >= threshold  # Include values equal to threshold

        # Use non-linear optimization to fit the Zipf function
        popt, _ = curve_fit(zipf_func, ranks[high_freq_indices], frequencies[high_freq_indices], p0=[1.0, np.max(frequencies)], maxfev=10000)

        # popt contains the fitted parameters alpha and C
        return popt[0]

    def plot_zipfs_law_fit(self, corpus_name):
        # Use the calculate_alpha method to obtain alpha
        alpha = self.calculate_alpha()

        ranks = np.arange(1, len(self.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.frequency.most_common()])
        
        # Normalize the actual frequencies
        normalized_frequencies = frequencies / np.max(frequencies)
        
        # Predict frequencies using the fitted alpha
        predicted_freqs = (1 / ranks**alpha)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, normalized_frequencies, color='blue', label='Actual Frequencies', marker='o', linestyle='', s=5)
        plt.plot(ranks, normalized_predicted_freqs, label='Predicted Frequencies (Zipf\'s Law)', color='red', linestyle='-')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf\'s Law Fit for {corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

    def fit_zipf_mandelbrot(self, initial_params=None, method='Nelder-Mead', verbose=False):
        """
        Fit the Zipf-Mandelbrot distribution to the corpus and find parameters q and s.
        """
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])
        ranks = np.array([details['rank'] for details in self.token_details.values()])

        # Normalizing data
        max_freq = frequencies.max()
        normalized_freqs = frequencies / max_freq

        def zipf_mandelbrot(k, q, s):
            return (1 / ((k + q) ** s))

        def objective_function(params):
            q, s = params
            predicted = np.array([zipf_mandelbrot(rank, q, s) for rank in ranks])
            normalized_predicted = predicted / predicted.max()  # Normalize predictions
            return np.sum((normalized_freqs - normalized_predicted) ** 2)

        # Adaptive initial parameters if not provided
        if initial_params is None:
            initial_params = [2.7, 1.0]  # Empirical initial values

        # Adjusting bounds based on empirical data
        bounds = [(0, 10), (0.5, 3)]

        # Optimization to minimize the objective function
        result = minimize(objective_function, initial_params, method=method, bounds=bounds, options={'disp': verbose})

        if result.success:
            q, s = result.x
            if verbose:
                print(f"Optimization successful. Fitted parameters: q = {q}, s = {s}")
            return q, s
        else:
            if verbose:
                print("Optimization did not converge.")
            raise ValueError("Optimization did not converge")

    def plot_zipf_mandelbrot_fit(self, q, s, corpus_name):
        """
        Plot the actual vs predicted rank-frequency distribution using the Zipf-Mandelbrot model.
        
        :param q: Fitted parameter q of the Zipf-Mandelbrot distribution.
        :param s: Fitted parameter s of the Zipf-Mandelbrot distribution.
        :param corpus_name: Name of the corpus for labeling the plot.
        """
        ranks = np.array([details['rank'] for details in self.token_details.values()])
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])

        # Zipf-Mandelbrot function
        def zipf_mandelbrot(k, q, s):
            return (1 / ((k + q) ** s))

        # Predict frequencies using the fitted parameters
        predicted_freqs = np.array([zipf_mandelbrot(rank, q, s) for rank in ranks])

        # Normalize both actual and predicted frequencies for comparison
        max_freq = np.max(frequencies)
        normalized_freqs = frequencies / max_freq
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, normalized_freqs, label='Actual Frequencies', marker='o', linestyle='', markersize=5)
        plt.plot(ranks, normalized_predicted_freqs, label='Predicted Frequencies', linestyle='-', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf-Mandelbrot Fit for {corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

class ZipfianAnalysis(BasicCorpusAnalyzer):
    def __init__(self, tokens):
        super().__init__(tokens)

    @staticmethod
    @lru_cache(maxsize=None)
    def _calculate_generalized_harmonic(n, alpha) -> float:
        """Calculate the generalized harmonic number."""
        return sum(1 / (i ** alpha) for i in range(1, n + 1))

    def plot_zipfian_comparison(self, alpha=1):
        """Plot the corpus frequency distribution against the ideal Zipfian distribution curve."""
        most_common = self.frequency.most_common()
        ranks = np.arange(1, len(most_common) + 1)
        frequencies = np.array([freq for _, freq in most_common])
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        harmonic_number = self._calculate_generalized_harmonic(len(ranks), alpha)
        ideal_zipfian = (self.total_token_count / harmonic_number) / np.power(ranks, alpha)
        log_ideal_zipfian = np.log(ideal_zipfian)

        plt.figure(figsize=(10, 6))
        plt.scatter(log_ranks, log_freqs, color='blue', label='Actual Frequencies')
        plt.plot(log_ranks, log_ideal_zipfian, 'g--', label='Ideal Zipfian')
        plt.xlabel('Log of Rank')
        plt.ylabel('Log of Frequency')
        plt.title('Zipfian Comparison of Corpus')
        plt.legend()
        plt.show()

    def assess_zipfian_fit(self, alpha=1) -> tuple:
        """Assess the fit of the corpus to a Zipfian distribution."""
        most_common = self.frequency.most_common()
        n = len(most_common)
        ranks = np.arange(1, n + 1)
        harmonic_number = ZipfianAnalysis._calculate_generalized_harmonic(n, alpha)
        harmonic_factor = self.total_token_count / harmonic_number
        expected_freqs = harmonic_factor / np.power(ranks, alpha)
        actual_freqs = np.array([freq for _, freq in most_common])

        deviations = actual_freqs - expected_freqs
        mean_deviation = np.mean(np.abs(deviations))
        std_deviation = np.std(deviations)
        return mean_deviation, std_deviation


