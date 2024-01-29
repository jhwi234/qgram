# Standard library imports
import os
import string
import math
from collections import Counter
import regex as re
import sys

# Third-party imports
import nltk
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Directory for plots
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

class CorpusLoader:
    """
    Load a corpus from NLTK or a local file/directory, optimized for performance.
    """
    def __init__(self, corpus_source, allow_download=True, custom_download_dir=None):
        # Source of the corpus (either a local path or an NLTK corpus name)
        self.corpus_source = corpus_source
        # Flag to allow automatic download from NLTK if the corpus isn't available locally
        self.allow_download = allow_download
        # Custom directory for downloading corpora (if needed)
        self.custom_download_dir = custom_download_dir
        # Cache for loaded corpus to avoid reloading
        self.corpus_cache = None

    def _download_corpus(self):
        """Download the corpus from NLTK."""
        # Append custom download directory to NLTK's data path if provided
        if self.custom_download_dir:
            nltk.data.path.append(self.custom_download_dir)
        # Download corpus using NLTK's download utility
        nltk.download(self.corpus_source, download_dir=self.custom_download_dir, quiet=True)

    def _load_corpus(self):
        """Load the corpus into memory."""
        # Handle loading from a local file or directory
        if os.path.isfile(self.corpus_source) or os.path.isdir(self.corpus_source):
            corpus_reader = PlaintextCorpusReader(self.corpus_source, '.*')
        else:
            # Access corpus from NLTK if it's a named dataset
            corpus_reader = getattr(nltk.corpus, self.corpus_source)
        # Read all tokens from the corpus
        return [token for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]

    def load_corpus(self):
        """Get the corpus, either from cache or by loading it."""
        # Return cached corpus if available
        if self.corpus_cache is not None:
            return self.corpus_cache

        # Download corpus if not locally available and downloading is allowed
        if not self.is_corpus_available() and self.allow_download:
            self._download_corpus()

        # Load corpus into cache and return
        self.corpus_cache = self._load_corpus()
        return self.corpus_cache

    def is_corpus_available(self):
        """Check if the corpus is available locally or through NLTK."""
        try:
            # Check if the corpus can be found by NLTK
            nltk.data.find(self.corpus_source)
            return True
        except LookupError:
            return False
class Tokenizer:
    """
    Tokenize text into individual words.
    """
    def __init__(self, remove_stopwords=False, remove_punctuation=False, use_nltk_tokenizer=True, stopwords_language='english'):
        # Configuration options for tokenization
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.stopwords_language = stopwords_language
        self.custom_regex = None
        # Set to store unwanted tokens (stopwords, punctuation) for removal
        self._unwanted_tokens = set()
        # Ensure necessary NLTK resources are available
        self._ensure_nltk_resources()
        # Load stopwords and punctuation into the set
        self._load_unwanted_tokens()

    def _ensure_nltk_resources(self):
        """Ensure that NLTK resources are available."""
        if self.remove_stopwords:
            # Download NLTK stopwords if they are not already available
            try:
                nltk.data.find(f'corpora/stopwords/{self.stopwords_language}')
            except LookupError:
                nltk.download('stopwords', quiet=True)

    def _load_unwanted_tokens(self):
        """Load stopwords and punctuation sets for efficient access."""
        if self.remove_stopwords:
            # Update the set with stopwords for the specified language
            self._unwanted_tokens.update(stopwords.words(self.stopwords_language))
        if self.remove_punctuation:
            # Update the set with string punctuation characters
            self._unwanted_tokens.update(string.punctuation)

    def set_custom_regex(self, pattern):
        """Set a custom regex pattern for tokenization with caching."""
        # Compile regex pattern for custom tokenization
        try:
            self.custom_regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def _remove_unwanted_tokens(self, tokens) -> list:
        """Remove unwanted tokens (stopwords, punctuation) from a list of tokens."""
        # Filter out tokens present in the unwanted tokens set
        return [token for token in tokens if token not in self._unwanted_tokens and not token.startswith('``')]
    
    def tokenize(self, text, lowercase=False) -> list:
        """Tokenize text into individual words based on the selected method."""
        if isinstance(text, list):
            # If input is a list, join it into a single string
            text = ' '.join(text)

        if lowercase:
            # Convert text to lowercase if specified
            text = text.lower()

        # Perform tokenization based on the selected method
        if self.custom_regex:
            # Tokenization using the custom regex pattern
            tokens = self.custom_regex.findall(text)
        elif self.use_nltk_tokenizer:
            # Tokenization using NLTK's word tokenizer
            tokens = word_tokenize(text)
        else:
            # Basic whitespace tokenization
            tokens = text.split()

        # Remove unwanted tokens from the result
        return self._remove_unwanted_tokens(tokens)
class CorpusTools:
    """
    Analyze a corpus of text with optimized data structures.
    Provides basic corpus analysis tools like frequency distribution and queries.
    """

    def __init__(self, tokens, shuffle_tokens=False):
        # Ensure all items in tokens are strings
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("All tokens must be strings.")
        
        self.tokens = tokens
        # Shuffle option for randomizing order, useful in certain analyses to remove bias
        if shuffle_tokens:
            self._shuffle_tokens()

        # Frequency distribution of tokens
        self.frequency = Counter(self.tokens)
        # Total token count to avoid repeated calculations
        self._total_token_count = sum(self.frequency.values())
        # Token details including frequency and rank
        self.token_details = self._initialize_token_details()

    def _shuffle_tokens(self):
        """Shuffle the tokens in the corpus using NumPy for efficiency."""
        np.random.shuffle(self.tokens)

    def _initialize_token_details(self):
        """
        Initialize a dictionary with token details including frequency, rank, and cumulative frequency.
        Useful for quick lookups and analysis.
        """
        token_details = {}
        cumulative = 0
        for rank, (token, freq) in enumerate(self.frequency.most_common(), start=1):
            # Cumulative frequency helps in calculating median and other stats
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
        """
        Find the median token based on frequency. 
        The median token is the token in the middle of the sorted frequency distribution.
        """
        median_index = self._total_token_count / 2
        cumulative = 0
        for token, details in self.token_details.items():
            cumulative += details['frequency']
            if cumulative >= median_index:
                # Token that crosses the median count is the median token
                return {'token': token, 'frequency': details['frequency']}

    def mean_token_frequency(self) -> float:
        """Calculate the mean frequency of tokens across the corpus."""
        return self.total_token_count / len(self.token_details)

    def query_by_token(self, token) -> dict:
        """Retrieve frequency and rank details for a specific token."""
        token = token.lower()
        details = self.token_details.get(token)
        if details:
            return {'token': token, **details}
        else:
            raise ValueError(f"Token '{token}' not found in the corpus.")

    def query_by_rank(self, rank) -> dict:
        """Retrieve token and frequency details for a specific rank in the frequency distribution."""
        for token, details in self.token_details.items():
            if details['rank'] == rank:
                return {'token': token, 'frequency': details['frequency'], 'rank': rank}
        raise ValueError(f"Rank {rank} is out of range.")
    
    def cumulative_frequency_analysis(self, lower_percent=0, upper_percent=100):
        """
        Analyze tokens within a specific cumulative frequency range. 
        Useful for understanding distribution of common vs. rare tokens.
        """
        # Validation of percentage inputs
        if not 0 <= lower_percent <= 100 or not 0 <= upper_percent <= 100:
            raise ValueError("Percentages must be between 0 and 100.")

        # Adjusting the range if inputs are in reverse order
        if lower_percent > upper_percent:
            lower_percent, upper_percent = upper_percent, lower_percent

        # Calculate threshold counts based on percentages
        lower_threshold = self._total_token_count * (lower_percent / 100)
        upper_threshold = self._total_token_count * (upper_percent / 100)

        # Extract tokens within the specified cumulative frequency range
        return [{'token': token, **details}
                for token, details in self.token_details.items()
                if lower_threshold <= details['cumulative_freq'] <= upper_threshold]

    def list_tokens_in_rank_range(self, start_rank, end_rank):
        """
        List tokens within a specific rank range. 
        Useful for examining the most/least frequent subsets of tokens.
        """
        # Validate rank range inputs
        if not (1 <= start_rank <= end_rank <= len(self.token_details)):
            raise ValueError("Rank range is out of valid bounds.")

        # Extract tokens within the specified rank range
        return [{'token': token, **details}
                for token, details in self.token_details.items()
                if start_rank <= details['rank'] <= end_rank]
    
    def x_legomena_count(self, x) -> int:
        """
        Count the number of x-legomena (tokens that occur x times) in the corpus.
        """
        return sum(1 for _, details in self.token_details.items() if details['frequency'] == x)
    
    def hapax_legomena_count(self) -> int:
        """Count the number of hapax legomena (tokens that occur only once) in the corpus."""
        return self.x_legomena_count(1)
    
    def vocabulary(self) -> set:
        """Return the set of distinct tokens in the corpus."""
        return set(self.frequency.keys())
    
    def vocabulary_size(self) -> int:
        """Return the number of distinct tokens in the corpus."""
        return len(self.frequency)

class AdvancedTools(CorpusTools):
    """
    Advanced analysis on a corpus of text, extending CorpusTools.
    Implements advanced linguistic metrics and statistical law calculations.
    """

    def __init__(self, tokens):
        super().__init__(tokens)
        # Caches to store precomputed parameters for efficient reuse
        self._zipf_params = None  # For Zipf's Law
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
        V = self.vocabulary_size()  # Distinct token count (Vocabulary size)
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

        # Adjust the sampling rate based on Herdan's C value to ensure a diverse range of samples
        base_sampling_rate = 0.05 + 0.05 * herdans_c_value
        base_sampling_rate = min(base_sampling_rate, 0.1)  # Set a reasonable upper bound for the sampling rate
        base_num_samples = 1000 + int(500 * herdans_c_value)

        # Determine the sizes of corpus samples to be analyzed
        adjusted_num_samples = min(base_num_samples, int(corpus_size * base_sampling_rate))
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

    def calculate_zipf_params(self):
        """
        Calculate parameters for Zipf's Law, relating word frequency and rank.
        """
        if self._zipf_params is not None:
            return self._zipf_params  # Use cached parameters if available

        # Normalize frequencies to compare with theoretical distribution
        ranks = np.arange(1, len(self.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.frequency.most_common()])
        total_tokens = sum(frequencies)
        normalized_frequencies = frequencies / total_tokens

        # Objective function aims to fit the corpus data to the Zipfian distribution
        def objective_function(params):
            alpha, c = params
            # Predicted frequencies according to Zipf's Law
            predicted_freqs = c / np.power(ranks, alpha)
            # Minimization target is the sum of squared differences
            return np.sum((normalized_frequencies - predicted_freqs) ** 2)

        initial_c = np.median(frequencies) / total_tokens
        initial_guess = [-1.0, initial_c]

        # Nonlinear optimization to find best-fit parameters
        result = minimize(objective_function, initial_guess, method='Nelder-Mead')

        if result.success:
            alpha, c = result.x
            self._zipf_params = (alpha, c)
        else:
            self._zipf_params = (None, None) # Mark as failed
            raise RuntimeError("Optimization failed to converge")

        return self._zipf_params

    def assess_zipf_fit(self):
        """Calculate the mean and standard deviation of residuals to assess the Zipfian distribution fit."""
        if self._zipf_params is None or self._zipf_params[0] is None:
            raise ValueError("Alpha and c have not been calculated or optimization failed.")
        
        alpha, c = self._zipf_params
        # Normalizing frequencies and preparing data
        total_count = sum(self.frequency.values())
        sorted_freqs = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        normalized_freqs = {word: freq / total_count for word, freq in sorted_freqs}

        # Compute residuals
        residuals = []
        for index, (word, freq) in enumerate(sorted_freqs, start=1): # Start at 1 to avoid division by zero
            rank = index 
            expected_freq = c / (rank ** alpha) # Zipf's Law
            residual = normalized_freqs[word] - expected_freq # Difference between actual and expected frequencies
            residuals.append(residual)

        # Calculate mean and standard deviation of residuals
        mean_residual = np.mean(residuals)
        std_dev_residual = np.std(residuals)
        return mean_residual, std_dev_residual

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
        bounds = [(1, 10), (0.1, 3)]

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
        Calculate the mean and standard deviation of residuals to assess the Zipf-Mandelbrot distribution fit.
        """
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])
        ranks = np.array([details['rank'] for details in self.token_details.values()])

        # Normalizing actual frequencies
        max_freq = frequencies.max()
        normalized_freqs = frequencies / max_freq

        # Zipf-Mandelbrot function
        def zipf_mandelbrot(k, q, s):
            return (1 / ((k + q) ** s))

        # Compute predicted frequencies
        predicted_freqs = np.array([zipf_mandelbrot(rank, q, s) for rank in ranks])
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Compute residuals
        residuals = np.abs(normalized_freqs - normalized_predicted_freqs)

        # Calculate mean and standard deviation of residuals
        mean_residual = np.mean(residuals)
        std_dev_residual = np.std(residuals)
        return mean_residual, std_dev_residual
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
        Plots the rank-frequency distribution of the corpus using Zipf's Law.
        Illustrates the relationship between word ranks and their frequencies.
        """
        # Check if Zipf's Law parameters are already calculated
        if self.analyzer._zipf_params is None:
            alpha, c = self.analyzer.calculate_zipf_params()
        else:
            alpha, c = self.analyzer._zipf_params

        # Check if the calculation was successful
        if alpha is None or c is None:
            raise ValueError("Alpha or c calculation failed, cannot plot Zipf's Law fit.")

        # Generate ranks and retrieve frequencies
        ranks = np.arange(1, len(self.analyzer.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.analyzer.frequency.most_common()])

        # Normalize frequencies for plotting on a logarithmic scale
        normalized_frequencies = frequencies / np.max(frequencies)

        # Calculate predicted frequencies using Zipf's Law
        predicted_freqs = c / np.power(ranks, alpha)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Setting up the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, normalized_frequencies, color='blue', label='Actual Frequencies', marker='o', linestyle='', s=5)
        plt.plot(ranks, normalized_predicted_freqs, label=f'Zipf\'s Law Fit (alpha={alpha:.2f}, c={c:.2f})', color='red', linestyle='-')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf\'s Law Fit for {self.corpus_name} Corpus')
        plt.xscale('log')  # Logarithmic scale to visualize Zipf's Law
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'zipfs_law_fit_{self.corpus_name}.png'))
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