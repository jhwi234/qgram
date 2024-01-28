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
from matplotlib import style
from scipy.optimize import curve_fit
from scipy.optimize import minimize

# Directory for plots
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

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
        self.alpha = self.calculate_alpha()

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

    def estimate_vocabulary_size(self, total_tokens):
        """
        Estimate the vocabulary size for a given number of tokens using Heaps' Law.

        :param total_tokens: Total number of tokens in the corpus.
        :return: Estimated vocabulary size.
        """
        K, beta = self.calculate_heaps_law()
        estimated_vocab_size = K * (total_tokens ** beta)
        return estimated_vocab_size

    def calculate_heaps_law(self, use_sampling=True):
        """
        Calculate Heap's Law constants, with an option to use sampling or standard approach.
        
        :param use_sampling: If True, use the sampling approach; otherwise, use the standard approach.
        """
        if len(self.tokens) < 2:
            raise ValueError("Not enough data to calculate Heaps' law constants.")

        # Common setup for both approaches
        word_set = set()
        unique_word_counts = []
        token_index = 0

        if use_sampling:
            # Sampling approach
            corpus_size = len(self.tokens)
            sampling_rate = 0.05
            sample_points = max(int(corpus_size * sampling_rate), 1000)
            sample_sizes = np.unique(np.logspace(0, np.log10(corpus_size), sample_points).astype(int))
            
            for size in sample_sizes:
                while token_index < size and token_index < corpus_size:
                    word_set.add(self.tokens[token_index])
                    token_index += 1
                unique_word_counts.append(len(word_set))

            # Fit in log-log space
            log_sample_sizes = np.log(sample_sizes)
            weights = np.sqrt(sample_sizes)
            beta, logK = np.polyfit(log_sample_sizes, np.log(unique_word_counts), 1, w=weights)

        else:
            # Standard approach
            total_words = np.arange(1, len(self.tokens) + 1)

            for token in self.tokens:
                word_set.add(token)
                unique_word_counts.append(len(word_set))

            # Fit in log-log space
            beta, logK = np.polyfit(np.log(total_words), np.log(unique_word_counts), 1)

        K = np.exp(logK)
        return K, beta

    def estimate_k_and_beta(self):
        """
        Estimates the parameters k and beta for Heap's Law using linear regression in log-log space.
        """
        N = self.total_token_count
        V = len(self.frequency)

        # Log-log transformation
        log_N = np.log10(N)
        log_V = np.log10(V)

        # Fitting a line (logV = beta * logN + logK) in log-log space
        beta, logK = np.polyfit(log_N, log_V, 1)

        # Transform back K from log space
        K = 10 ** logK

        return K, beta

    @staticmethod
    @lru_cache(maxsize=None)
    def _calculate_generalized_harmonic(n, alpha) -> float:
        """Calculate the generalized harmonic number."""
        return sum(1 / (i ** alpha) for i in range(1, n + 1))

    def calculate_alpha(self):
        """
        Calculate the alpha parameter of Zipf's Law for the corpus.
        """
        # Extract ranks (1-based) and frequencies
        ranks = np.arange(1, len(self.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.frequency.most_common()])

        # Define the Zipf function for fitting
        def zipf_func(rank, alpha, c):
            return c / np.power(rank, alpha)

        # Fit the Zipf function using non-linear least squares
        popt, _ = curve_fit(zipf_func, ranks, frequencies, p0=[-1.0, np.max(frequencies)])
        
        # popt contains the fitted parameters alpha and c
        alpha = popt[0]
        return alpha

    def assess_alpha_fit(self, alpha):
        """Assess the fit of the corpus to a Zipfian distribution."""
        # Normalizing frequencies and preparing data
        total_count = sum(self.frequency.values())
        sorted_freqs = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        normalized_freqs = {word: freq / total_count for word, freq in sorted_freqs}

        # Handling ties in ranks
        current_rank, current_freq = 0, None
        deviations = []
        for index, (word, freq) in enumerate(sorted_freqs, start=1):
            if freq != current_freq:
                current_rank = index
                current_freq = freq

            expected_freq = (1 / current_rank ** self.alpha)
            deviations.append(abs(normalized_freqs[word] - expected_freq))

        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)

        return mean_deviation, std_deviation

    def calculate_zipf_mandelbrot(self, initial_params=None, method='Nelder-Mead', verbose=False):
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
        bounds = [(0, 15), (0.25, 3)]

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
        
    def assess_zipf_mandelbrot_fit(self, q, s):
        """
        Assess the fit of the Zipf-Mandelbrot distribution to the corpus using the provided parameters q and s.
        """
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])
        ranks = np.array([details['rank'] for details in self.token_details.values()])

        # Normalizing actual frequencies
        max_freq = frequencies.max()
        normalized_freqs = frequencies / max_freq

        # Zipf-Mandelbrot function
        def zipf_mandelbrot(k, q, s):
            return (1 / ((k + q) ** s))

        # Predict frequencies using the provided parameters
        predicted_freqs = np.array([zipf_mandelbrot(rank, q, s) for rank in ranks])
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Calculate deviations
        deviations = np.abs(normalized_freqs - normalized_predicted_freqs)
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)

        return mean_deviation, std_deviation
    
class CorpusPlotter:
    def __init__(self, analyzer, corpus_name, plots_dir='plots'):
        """
        Initializes the CorpusPlotter with an analyzer, corpus name, and plots directory.
        """
        self.analyzer = analyzer
        self.corpus_name = corpus_name
        self.plots_dir = plots_dir

    def plot_zipf_mandelbrot_fit(self, q, s):
        """
        Plots the fitted parameters of the Zipf-Mandelbrot distribution.
        """
        # Retrieve rank and frequency data from the analyzer
        ranks = np.array([details['rank'] for details in self.analyzer.token_details.values()])
        frequencies = np.array([details['frequency'] for details in self.analyzer.token_details.values()])

        # Define the Zipf-Mandelbrot function for frequency prediction
        def zipf_mandelbrot(k, q, s):
            return (1 / ((k + q) ** s))

        # Compute predicted frequencies
        predicted_freqs = np.array([zipf_mandelbrot(rank, q, s) for rank in ranks])

        # Normalize frequencies for plotting
        normalized_freqs = frequencies / np.max(frequencies)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Plotting setup
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, normalized_freqs, label='Actual Frequencies', marker='o', linestyle='', markersize=5)
        plt.plot(ranks, normalized_predicted_freqs, label='Predicted Frequencies', linestyle='-', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf-Mandelbrot Fit for {self.corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'zipf_mandelbrot_fit_{self.corpus_name}.png'))
        plt.close()

    def plot_zipfs_law_fit(self):
        """
        Plots the rank-frequency distribution of the corpus using Zipf's Law.
        """
        # Calculate alpha using the analyzer's method
        alpha = self.analyzer.calculate_alpha()

        # Prepare ranks and frequencies
        ranks = np.arange(1, len(self.analyzer.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.analyzer.frequency.most_common()])

        # Normalize frequencies and predict using Zipf's Law
        normalized_frequencies = frequencies / np.max(frequencies)
        predicted_freqs = (1 / ranks ** alpha)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Plot setup
        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, normalized_frequencies, color='blue', label='Actual Frequencies', marker='o', linestyle='', s=5)
        plt.plot(ranks, normalized_predicted_freqs, label='Predicted Frequencies (Zipf\'s Law)', color='red', linestyle='-')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf\'s Law Fit for {self.corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'zipfs_law_fit_{self.corpus_name}.png'))
        plt.close()

    def plot_heaps_law(self, K, beta):
        """
        Plots the relationship between the number of unique words and the total number of words in the corpus, illustrating Heap's Law.
        """
        # Prepare data for Heap's Law plot
        total_words = np.arange(1, len(self.analyzer.tokens) + 1)
        unique_words = []
        word_set = set()

        # Count unique words
        for token in self.analyzer.tokens:
            word_set.add(token)
            unique_words.append(len(word_set))

        # Plot setup
        plt.figure(figsize=(10, 6))
        plt.plot(total_words, unique_words, label='Empirical Data', color='blue')
        plt.plot(total_words, K * np.power(total_words, beta), '--', label=f'Heap\'s Law Fit: K={K:.2f}, beta={beta:.2f}', color='red')
        plt.xlabel('Token Count')
        plt.ylabel('Type Count')
        plt.title(f"Heap's Law Analysis for {self.corpus_name} Corpus")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'heaps_law_{self.corpus_name}.png'))
        plt.close()
