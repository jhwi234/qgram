# Standard library imports
import os
from typing import Dict, List, Set
import math
from collections import Counter
import regex as reg
import sys

# Third-party imports
import nltk
from nltk.corpus import PlaintextCorpusReader, LazyCorpusLoader, stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import zeta
import matplotlib.pyplot as plt

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
        # Use the first path in nltk.data.path as the default download directory if custom_download_dir is not provided.
        self.custom_download_dir = custom_download_dir or nltk.data.path[0]  
        self.corpus_cache = None

    def _download_corpus(self):
        if self.allow_download and not self.is_corpus_available():
            # Attempt to download the corpus to the specified custom directory or to the default NLTK data directory.
            nltk.download(self.corpus_source, download_dir=self.custom_download_dir)

    def _load_corpus(self):
        if os.path.exists(self.corpus_source):
            # Load the corpus from a local directory or file.
            corpus_reader = PlaintextCorpusReader(root=self.corpus_source, fileids='.*')
        else:
            # Lazy load the corpus from NLTK if it's a named dataset.
            # This approach avoids loading the entire corpus into memory at once.
            corpus_reader = LazyCorpusLoader(self.corpus_source, PlaintextCorpusReader, r'.*')
            if self.allow_download:
                self._download_corpus()  # Ensure the corpus is downloaded if not available locally.
        return corpus_reader

    def load_corpus(self):
        if not self.corpus_cache:
            self.corpus_cache = self._load_corpus()
        # Access corpus content directly. LazyCorpusLoader will handle loading as needed.
        return list(self.corpus_cache.words())

    def is_corpus_available(self):
        try:
            # Check if the corpus can be found locally or within NLTK's available corpora.
            nltk.data.find(self.corpus_source)
            return True
        except LookupError:
            return False
class Tokenizer:
    """
    Tokenize text into individual words, with options for removing stopwords and punctuation.
    """
    def __init__(self, remove_stopwords=False, remove_punctuation=False, stopwords_language='english'):
        # Configuration options for tokenization
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        
        # Initialize the set of stopwords if needed
        if remove_stopwords:
            nltk.download('stopwords', quiet=True)  # Ensure stopwords are downloaded
            self.stopwords_set = set(stopwords.words(stopwords_language))
        else:
            self.stopwords_set = set()
        
        # Select tokenizer method based on the need to remove punctuation
        if remove_punctuation:
            # Use RegexpTokenizer to exclude punctuation during tokenization
            self.tokenizer = RegexpTokenizer(r'\w+')
        else:
            # Use NLTK's default word tokenizer, which retains punctuation
            self.tokenizer = word_tokenize

    def tokenize(self, text, lowercase=False):
        """
        Tokenizes the text into words, optionally removes stopwords and/or punctuation, and optionally converts to lowercase.
        
        Parameters:
        - text: The text to be tokenized.
        - lowercase: Whether to convert the text to lowercase before tokenization.
        
        Returns:
        - A list of tokenized words.
        """
        # Optionally convert text to lowercase before tokenization
        text = text.lower() if lowercase else text
        
        # Tokenize the text
        tokens = self.tokenizer(text)
        
        # Optionally remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords_set]
        
        return tokens
class CorpusTools:
    """
    Provides basic corpus analysis tools like frequency distribution and querying capabilities, optimized with NumPy.
    """

    def __init__(self, tokens: List[str], shuffle_tokens: bool = False):
        """
        Initialize the CorpusTools object with a list of tokens, optionally shuffling them for unbiased analysis.

        :param tokens: A list of tokens (words) in the corpus.
        :param shuffle_tokens: Whether to shuffle the tokens. Defaults to False.
        """
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("All tokens must be strings.")

        if shuffle_tokens:
            np.random.shuffle(tokens)

        self.tokens = np.array(tokens)
        self.frequency = Counter(tokens)
        self.token_frequencies = np.array(list(self.frequency.values()))
        self.tokens_unique = np.array(list(self.frequency.keys()))
        indices_sorted = np.argsort(self.token_frequencies)[::-1]
        self.tokens_sorted = self.tokens_unique[indices_sorted]
        self.token_frequencies_sorted = self.token_frequencies[indices_sorted]
        self._total_token_count = np.sum(self.token_frequencies)

    @property
    def total_token_count(self) -> int:
        """Returns the total number of tokens in the corpus."""
        return self._total_token_count

    def find_median_token(self) -> Dict[str, int]:
        """
        Finds the median token based on frequency, which divides the corpus into two equal halves.

        :return: A dictionary with the median token and its frequency.
        """
        cumulative = np.cumsum(self.token_frequencies_sorted)
        median_index = np.searchsorted(cumulative, self._total_token_count / 2, side='right')
        return {'token': self.tokens_sorted[median_index], 'frequency': self.token_frequencies_sorted[median_index]}

    def mean_token_frequency(self) -> float:
        """
        Calculates the mean frequency of tokens in the corpus.

        :return: The mean token frequency as a float.
        """
        return np.mean(self.token_frequencies)

    def query_by_token(self, token: str) -> Dict[str, int]:
        """
        Retrieves frequency and rank details for a specific token.

        :param token: The token to query.
        :return: A dictionary with token details (frequency and rank).
        :raises ValueError: If the token is not found in the corpus.
        """
        token = token.lower()
        index = np.where(self.tokens_unique == token)[0]
        if len(index) == 0:
            raise ValueError(f"Token '{token}' not found in the corpus.")
        index = index[0]
        return {'token': token, 'frequency': self.frequency[token], 'rank': np.searchsorted(self.tokens_sorted, token, sorter=np.argsort(self.tokens_sorted)) + 1}

    def query_by_rank(self, rank: int) -> Dict[str, int]:
        """
        Retrieves token details for a specific rank in the frequency distribution.

        :param rank: The rank to query.
        :return: A dictionary with token details for the given rank.
        :raises ValueError: If the rank is out of range.
        """
        if rank < 1 or rank > len(self.tokens_unique):
            raise ValueError("Rank is out of range.")
        token = self.tokens_sorted[rank - 1]
        frequency = self.token_frequencies_sorted[rank - 1]
        return {'token': token, 'rank': rank, 'frequency': frequency}

    def cumulative_frequency_analysis(self, lower_percent: float = 0, upper_percent: float = 100) -> List[Dict[str, int]]:
        """
        Analyzes tokens within a specific cumulative frequency range.

        :param lower_percent: Lower bound of the cumulative frequency range (in percentage).
        :param upper_percent: Upper bound of the cumulative frequency range (in percentage).
        :return: A list of dictionaries with token details in the specified range.
        :raises ValueError: If the provided percentages are out of bounds.
        """
        cumulative = np.cumsum(self.token_frequencies_sorted) / self._total_token_count * 100
        indices = np.where((cumulative >= lower_percent) & (cumulative <= upper_percent))[0]
        return [{'token': self.tokens_sorted[i], 'frequency': self.token_frequencies_sorted[i]} for i in indices]

    def list_tokens_in_rank_range(self, start_rank: int, end_rank: int) -> List[Dict[str, int]]:
        """
        Lists tokens within a specific rank range.

        :param start_rank: Starting rank of the range.
        :param end_rank: Ending rank of the range.
        :return: A list of dictionaries with token details within the specified rank range.
        :raises ValueError: If the rank range is out of valid bounds.
        """
        if not (1 <= start_rank <= end_rank <= len(self.tokens_unique)):
            raise ValueError("Rank range is out of valid bounds.")
        return [{'token': self.tokens_sorted[i], 'frequency': self.token_frequencies_sorted[i]} for i in range(start_rank - 1, end_rank)]

    def x_legomena(self, x: int) -> Set[str]:
        """
        Lists tokens that occur exactly x times in the corpus.

        :param x: The number of occurrences to filter tokens by.
        :return: A set of tokens occurring exactly x times.
        :raises ValueError: If x is not a positive integer.
        """
        if not isinstance(x, int) or x < 1:
            raise ValueError("x must be a positive integer.")
        return {token for token, freq in zip(self.tokens_unique, self.token_frequencies) if freq == x}

    def vocabulary(self) -> Set[str]:
        """
        Returns the set of distinct tokens in the corpus.

        :return: A set of distinct tokens.
        """
        return set(self.tokens_unique)

class AdvancedTools(CorpusTools):
    """
    Advanced analysis on a corpus of text, extending CorpusTools.
    Implements advanced linguistic metrics and statistical law calculations.
    """

    def __init__(self, tokens):
        super().__init__(tokens)
        # Caches to store precomputed parameters for efficient reuse
        self._zipf_alpha = None  # For Zipf's Law
        self._zipf_c = None  # For Zipf's Law
        self._heaps_params = None  # For Heaps' Law
        self._zipf_mandelbrot_params = None  # For Zipf-Mandelbrot Law
        self.herdans_c_value = None  # For Herdan's C measure

    def yules_k(self) -> float:
        """
        Calculate Yule's K measure, indicating lexical diversity. Higher values suggest greater diversity.
        """
        # Using numpy for efficient array operations on token frequencies
        freqs = np.array([details['frequency'] for details in self.token_details.values()])
        N = np.sum(freqs)  # Total token count
        sum_fi_fi_minus_1 = np.sum(freqs * (freqs - 1))  # Sum of f_i * (f_i - 1) across all frequencies
        # Yule's K equation
        K = 10**4 * (sum_fi_fi_minus_1 / (N * (N - 1))) if N > 1 else 0
        return K

    def herdans_c(self) -> float:
        """
        Compute Herdan's C measure, reflecting vocabulary richness relative to corpus size.
        More comprehensive version of type-token ratio (TTR). Herdan's C is defined as the
        logarithm of the number of distinct tokens (V) divided by the logarithm of the total
        token count (N). It provides a normalized measure of vocabulary richness.
        Handles very large values of V and N by scaling them down to avoid precision issues.
        """
        if self.herdans_c_value is not None:
            return self.herdans_c_value  # Use cached value if available
        
        # Utilize properties from CorpusTools
        V = len(self.vocabulary())  # Distinct token count (Vocabulary size)
        N = self.total_token_count  # Total token count (Corpus size)

        # Check for edge cases to prevent division by zero or logarithm of zero
        if V == 0 or N == 0:
            raise ValueError("Vocabulary size (V) or total token count (N) cannot be zero.")

        # Handling very large values of V and N
        MAX_FLOAT = sys.float_info.max  # Maximum float value in the environment
        if V > MAX_FLOAT or N > MAX_FLOAT:
            # Apply scaling to reduce the values
            scaling_factor = max(V, N) / MAX_FLOAT
            V /= scaling_factor
            N /= scaling_factor

        # Calculating Herdan's C with error handling
        try:
            self.herdans_c_value = math.log(V) / math.log(N)
        except ValueError as e:
            # Handle potential math domain errors
            raise ValueError(f"Error in calculating Herdan's C: {e}")

        return self.herdans_c_value

    def calculate_heaps_law(self):
        """
        Estimate parameters for Heaps' Law, which predicts the growth of the number of distinct word types 
        in a corpus as the size of the corpus increases. This method adjusts the sampling of the corpus based 
        on its characteristics, determined by Herdan's C value, to accurately model vocabulary growth.
        """
        # Return cached Heaps' Law parameters if available
        if self._heaps_params is not None:
            return self._heaps_params

        # Use total_token_count from CorpusTools to determine the size of the corpus
        corpus_size = self.total_token_count

        # Herdan's C value is used to inform the sampling strategy, reflecting the complexity of the corpus
        herdans_c_value = self.herdans_c()

        # Calculate the sampling rate and number of samples based on Herdan's C value
        base_sampling_rate = min(0.05 + 0.05 * herdans_c_value, 0.1)
        adjusted_num_samples = min(1000 + int(500 * herdans_c_value), int(corpus_size * base_sampling_rate))

        # Determine the sizes of corpus samples to be analyzed
        sample_sizes = np.unique(np.linspace(0, corpus_size, num=adjusted_num_samples, endpoint=False).astype(int))

        # Set up data structures for calculating Heaps' Law
        distinct_word_types_set = set()
        distinct_word_counts = []

        # Count the number of distinct word types in each sample to model vocabulary growth
        for size in sample_sizes:
            sample_tokens = self.tokens[:size]
            distinct_word_types_set.update(sample_tokens)
            distinct_word_counts.append(len(distinct_word_types_set))

        # Apply linear regression to log-transformed sample sizes and distinct word counts for initial parameter estimation
        log_sample_sizes = np.log(sample_sizes[1:])  # Log-transform to linearize the growth relationship
        log_distinct_word_counts = np.log(distinct_word_counts[1:])
        beta, logK = np.polyfit(log_sample_sizes, log_distinct_word_counts, 1)
        K_linear = np.exp(logK)  # Convert back from log scale to obtain the initial estimate of K

        # Refine the estimates of K and beta using nonlinear optimization
        def objective_function(params):
            K, beta = params
            # Objective: Minimize the sum of squared differences between observed and predicted counts of distinct word types
            return np.sum((K * sample_sizes**beta - distinct_word_counts)**2)

        initial_params = [K_linear, beta]
        result = minimize(objective_function, initial_params, method='Nelder-Mead')

        # Use the results from linear regression if the optimization does not succeed
        K, beta = result.x if result.success else (K_linear, beta)

        # Cache the optimized parameters for future reference
        self._heaps_params = (K, beta)
        return K, beta
    
    def estimate_vocabulary_size(self, total_tokens) -> int:
        """
        Estimate the vocabulary size for a given number of tokens using Heaps' Law.
        Ensures that the output is an integer, as vocabulary size cannot be fractional.
        Use to evaluate Heaps' law fit by comparing with actual vocabulary size.
        """
        if self._heaps_params is None:
            self._heaps_params = self.calculate_heaps_law() # Calculate parameters if not already cached

        K, beta = self._heaps_params # Retrieve parameters
        estimated_vocab_size = K * (total_tokens ** beta) # Calculate vocabulary size

        # Round the estimated vocabulary size to the nearest integer
        return int(round(estimated_vocab_size))

    def calculate_zipf_alpha(self):
        """
        Calculate the alpha parameter for Zipf's Law using curve fitting with optimization.
        """
        if self._zipf_alpha is not None:
            # Use cached value if available
            return self._zipf_alpha

        # Define the Zipf function for curve fitting
        def zipf_func(rank, alpha, C):
            return C / np.power(rank, alpha)

        # Extract ranks and frequencies
        ranks = np.arange(1, len(self.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.frequency.most_common()])

        # Set up grid search for initial alpha guesses
        alpha_guesses = np.linspace(0.5, 1.25, num=1000)  # Adjust the range and number of points as needed
        best_alpha = None
        min_error = float('inf')

        # Loop over alpha guesses to find the best starting point
        for alpha_guess in alpha_guesses:
            try:
                popt, _ = curve_fit(zipf_func, ranks, frequencies, p0=[alpha_guess, np.max(frequencies)], maxfev=10000)
                current_error = np.sum(np.abs(frequencies - zipf_func(ranks, *popt)))
                if current_error < min_error:
                    min_error = current_error
                    best_alpha = popt[0]
            except RuntimeError:
                # Handle cases where curve_fit fails to converge for a given alpha guess
                continue

        if best_alpha is None:
            raise RuntimeError("Optimization failed to converge")

        # Cache the calculated alpha value
        self._zipf_alpha = best_alpha

        return best_alpha

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
        bounds = [(1.0, 10.0), (0.1, 3.0)]

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
class CorpusPlots:
    def __init__(self, analyzer, corpus_name, plots_dir='plots'):
        """
        Initializes the CorpusPlotter with an analyzer, corpus name, and plots directory.
        - analyzer: An instance of AdvancedTools or CorpusTools used for data analysis.
        - corpus_name: Name of the corpus, used for labeling plots.
        - plots_dir: Directory to save generated plots.
        """
        self.analyzer = analyzer
        self.corpus_name = corpus_name
        self.plots_dir = plots_dir

    def plot_zipfs_law_fit(self):
        """
        Plot the rank-frequency distribution using Zipf's Law focusing on alpha.
        """
        # Check if alpha is already calculated
        if self.analyzer._zipf_alpha is None:
            alpha = self.analyzer.calculate_zipf_alpha()
        else:
            alpha = self.analyzer._zipf_alpha
        
        # Check if the calculation was successful
        if alpha is None:
            raise ValueError("Alpha calculation failed, cannot plot Zipf's Law fit.")

        ranks = np.arange(1, len(self.analyzer.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.analyzer.frequency.most_common()])

        normalized_frequencies = frequencies / np.max(frequencies)
        predicted_freqs = 1 / np.power(ranks, alpha)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, normalized_frequencies, color='blue', label='Actual Frequencies', marker='o', linestyle='', s=5)
        plt.plot(ranks, normalized_predicted_freqs, label=f'Zipf\'s Law Fit (alpha={alpha:.2f})', color='red', linestyle='-')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf\'s Law Fit (Alpha Only) for {self.corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'zipfs_alpha_fit_{self.corpus_name}.png'))
        plt.close()

    def plot_heaps_law(self):
        """
        Plots the relationship between the number of unique words (types) and the total number of words (tokens) in the corpus, illustrating Heaps' Law.
        Demonstrates corpus vocabulary growth.
        """
        # Check if Heaps' Law parameters are already calculated
        if self.analyzer._heaps_params is None:
            K, beta = self.analyzer.calculate_heaps_law()
        else:
            K, beta = self.analyzer._heaps_params
        
        # Check is the calculation was successful
        if K is None or beta is None:
            raise ValueError("K or beta calculation failed, cannot plot Heaps' Law.")

        # Prepare data for plotting Heaps' Law
        total_words = np.arange(1, len(self.analyzer.tokens) + 1)
        unique_words = []
        word_set = set()

        # Counting unique words (types) as the corpus grows
        for token in self.analyzer.tokens:
            word_set.add(token)
            unique_words.append(len(word_set))

        # Plotting the empirical data and Heaps' Law fit
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
        This distribution is a generalization of Zipf's Law, adding parameters to account for corpus-specific characteristics.
        """
        # Check if Zipf-Mandelbrot parameters are already calculated
        if self.analyzer._zipf_mandelbrot_params is None:
            q, s = self.analyzer.calculate_zipf_mandelbrot()
        else:
            q, s = self.analyzer._zipf_mandelbrot_params

        # Check if the calculation was successful
        if q is None or s is None:
            raise ValueError("q or s calculation failed, cannot plot Zipf-Mandelbrot distribution.")
        
        ranks = np.array([details['rank'] for details in self.analyzer.token_details.values()])
        frequencies = np.array([details['frequency'] for details in self.analyzer.token_details.values()])

        # Defining the Zipf-Mandelbrot function
        def zipf_mandelbrot(k, q, s):
            return (1 / ((k + q) ** s))

        # Computing predicted frequencies using the Zipf-Mandelbrot parameters
        predicted_freqs = np.array([zipf_mandelbrot(rank, q, s) for rank in ranks])

        # Normalizing for plotting
        normalized_freqs = frequencies / np.max(frequencies)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Plotting the empirical data against the fitted distribution
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, normalized_freqs, label='Actual Frequencies', marker='o', linestyle='', markersize=5)
        plt.plot(ranks, normalized_predicted_freqs, label=f'Zipf-Mandelbrot Fit (q={q:.2f}, s={s:.2f})', linestyle='-', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf-Mandelbrot Fit for {self.corpus_name} Corpus')
        plt.xscale('log')  # Logarithmic scale for better visualization
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'zipf_mandelbrot_fit_{self.corpus_name}.png'))
        plt.close()