# Standard library imports
import os
import string
import math
from collections import Counter
from functools import lru_cache
import regex as re
import sys

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
class CorpusTools:
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
class AdvancedTools(CorpusTools):
    """
    Advanced analysis on a corpus of text, extending CorpusTools.
    This class includes advanced linguistic metrics such as Yule's K measure,
    Herdan's C measure, and calculations related to Heaps' Law.
    """
    def __init__(self, tokens):
        super().__init__(tokens)
        self._alpha_params = None # Cache for Zipf's Law alpha parameter
        self._heaps_params = None # Cache for Heaps' Law parameters
        self._zipf_mandelbrot_params = None # Cache for Zipf-Mandelbrot parameters
        self.herdans_c_value = None # Cache for Herdan's C value

    @staticmethod
    @lru_cache(maxsize=None)
    def _calculate_generalized_harmonic(n, alpha) -> float:
        """Calculate the generalized harmonic number."""
        return sum(1 / (i ** alpha) for i in range(1, n + 1))

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
        if self.herdans_c_value is not None:
            return self.herdans_c_value
        
        V = len(self.token_details)
        N = self._total_token_count
        self.herdans_c_value = math.log(V) / math.log(N)

        return self.herdans_c_value

    def calculate_heaps_law(self):
        """
        Estimate the parameters K and beta for Heaps' Law using a sampling approach.
        Dynamically adjust the sampling based on corpus vocabulary richness (Herdan's C).
        """
        if self._heaps_params is not None:
            return self._heaps_params
        
        corpus_size = len(self.tokens)
        
        # Directly using Herdan's C method from the same class
        herdans_c_value = self.herdans_c()

        # Base sampling rate and number of samples determined dynamically
        base_sampling_rate = 0.05 + 0.05 * herdans_c_value  # Adjust base rate based on Herdan's C
        base_sampling_rate = min(base_sampling_rate, 0.1)  # Limit the maximum sampling rate

        base_num_samples = 1000 + int(500 * herdans_c_value)  # Adjust number of samples based on Herdan's C

        # Determine sample sizes
        adjusted_num_samples = min(base_num_samples, int(corpus_size * base_sampling_rate))
        sample_sizes = np.unique(np.linspace(0, corpus_size, num=adjusted_num_samples, endpoint=False).astype(int))

        word_set = set()
        unique_word_counts = []

        for size in sample_sizes:
            sample_tokens = self.tokens[:size]
            word_set.update(sample_tokens)
            unique_word_counts.append(len(word_set))

        # Linear regression on log-transformed sample data
        log_sample_sizes = np.log(sample_sizes[1:])  # Exclude the first sample (size 0)
        log_unique_word_counts = np.log(unique_word_counts[1:])
        beta, logK = np.polyfit(log_sample_sizes, log_unique_word_counts, 1)
        K_linear = np.exp(logK)

        # Nonlinear optimization
        def objective_function(params):
            K, beta = params
            return np.sum((K * sample_sizes**beta - unique_word_counts)**2)

        initial_params = [K_linear, beta]
        result = minimize(objective_function, initial_params, method='Nelder-Mead')

        if result.success:
            K, beta = result.x
        else:
            K, beta = K_linear, beta  # Fall back to linear regression estimates

        # Cache the results before returning
        self._heaps_params = (K, beta)
        return K, beta

    def estimate_vocabulary_size(self, total_tokens):
        """
        Estimate the vocabulary size for a given number of tokens using Heaps' Law.

        :param total_tokens: Total number of tokens in the corpus.
        :return: Estimated vocabulary size.
        """
        if self._heaps_params is None:
            self._heaps_params = self.calculate_heaps_law()

        K, beta = self._heaps_params
        estimated_vocab_size = K * (total_tokens ** beta)
        return estimated_vocab_size

    def calculate_alpha(self):
        """
        Calculate the alpha parameter of Zipf's Law for the corpus using optimization.
        """
        if self._alpha_params is not None:
            return self._alpha_params
        
        # Extract ranks (1-based) and frequencies
        ranks = np.arange(1, len(self.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.frequency.most_common()])

        # Define the objective function for minimization
        def objective_function(params):
            alpha, c = params
            predicted_freqs = c / np.power(ranks, alpha)
            return np.sum((frequencies - predicted_freqs) ** 2)

        # Initial guesses for alpha and c
        initial_guess = [-1.0, np.max(frequencies)]

        # Optimization using a suitable method, e.g., Nelder-Mead
        result = minimize(objective_function, initial_guess, method='Nelder-Mead')

        if result.success:
            alpha, c = result.x
            self._alpha_params = (alpha, c)
        else:
            self._alpha_params = (None, None)  # Indicate failure
            raise RuntimeError("Optimization failed to converge")
        
        return self._alpha_params

    def assess_alpha_fit(self):
        """Assess the fit of the corpus to a Zipfian distribution considering both alpha and c."""
        if self._alpha_params is None or self._alpha_params[0] is None:
            raise ValueError("Alpha and c have not been calculated or optimization failed.")
        
        alpha, c = self._alpha_params
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

            expected_freq = c / (current_rank ** alpha)
            deviations.append(abs(normalized_freqs[word] - expected_freq))

        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)

        return mean_deviation, std_deviation

    def calculate_zipf_mandelbrot(self, initial_params=None, verbose=False):
        """
        Fit the Zipf-Mandelbrot distribution to the corpus and find parameters q and s.
        """
        if self._zipf_mandelbrot_params is not None:
            return self._zipf_mandelbrot_params
        
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])
        ranks = np.array([details['rank'] for details in self.token_details.values()])

        # Normalizing data
        max_freq = frequencies.max()
        normalized_freqs = frequencies / max_freq

        def zipf_mandelbrot_vectorized(ranks, q, s):
            return 1 / np.power(ranks + q, s)

        def objective_function(params):
            q, s = params
            predicted = zipf_mandelbrot_vectorized(ranks, q, s)
            normalized_predicted = predicted / np.max(predicted)
            return np.sum((normalized_freqs - normalized_predicted) ** 2)

        # Adaptive initial parameters if not provided
        if initial_params is None:
            initial_params = [2.7, 1.0]  # Empirical initial values

        # Adjusting bounds based on empirical data
        bounds = [(0, 10), (0, 10)]

        # Optimization to minimize the objective function
        result = minimize(objective_function, initial_params, method='Nelder-Mead', bounds=bounds, options={'disp': verbose})

        if result.success:
            q, s = result.x
            if verbose:
                print(f"Optimization successful. Fitted parameters: q = {q}, s = {s}")
            self._zipf_mandelbrot_params = q, s
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
    
class CorpusPlots:
    def __init__(self, analyzer, corpus_name, plots_dir='plots'):
        """
        Initializes the CorpusPlotter with an analyzer, corpus name, and plots directory.
        """
        self.analyzer = analyzer
        self.corpus_name = corpus_name
        self.plots_dir = plots_dir

    def plot_zipfs_law_fit(self):
        """
        Plots the rank-frequency distribution of the corpus using Zipf's Law.
        """
        alpha, c = self.analyzer.calculate_alpha()
        if alpha is None or c is None:
            raise ValueError("Alpha or c calculation failed, cannot plot Zipf's Law fit.")

        # Prepare ranks and frequencies
        ranks = np.arange(1, len(self.analyzer.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.analyzer.frequency.most_common()])

        # Normalize frequencies
        normalized_frequencies = frequencies / np.max(frequencies)

        # Predict frequencies using Zipf's Law
        predicted_freqs = c / np.power(ranks, alpha)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Plot setup
        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, normalized_frequencies, color='blue', label='Actual Frequencies', marker='o', linestyle='', s=5)
        plt.plot(ranks, normalized_predicted_freqs, label=f'Zipf\'s Law Fit (alpha={alpha:.2f}, c={c:.2f})', color='red', linestyle='-')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf\'s Law Fit for {self.corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'zipfs_law_fit_{self.corpus_name}.png'))
        plt.close()

    def plot_heaps_law(self):
        """
        Plots the relationship between the number of unique words and the total number of words in the corpus, illustrating Heap's Law.
        """
        # Directly use the cached values or calculate if not available
        K, beta = self.analyzer.calculate_heaps_law()

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
        plt.plot(total_words, K * np.power(total_words, beta), '--', 
                 label=f"Heap's Law Fit: K={K:.2f}, beta={beta:.2f}", color='red')
        plt.xlabel('Token Count')
        plt.ylabel('Type Count')
        plt.title(f"Heap's Law Analysis for {self.corpus_name} Corpus")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'heaps_law_{self.corpus_name}.png'))
        plt.close()

    def plot_zipf_mandelbrot_fit(self):
        """
        Plots the fitted parameters of the Zipf-Mandelbrot distribution.
        """
        q, s = self.analyzer.calculate_zipf_mandelbrot()
        
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
        plt.plot(ranks, normalized_predicted_freqs, label=f'Zipf-Mandelbrot Fit (q={q:.2f}, s={s:.2f})', linestyle='-', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf-Mandelbrot Fit for {self.corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'zipf_mandelbrot_fit_{self.corpus_name}.png'))
        plt.close()