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
    """
    Advanced corpus analysis.
    """
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
        """Calculate Yule's K and interpret the value."""
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
        """Calculate Herdan's C and interpret the value."""
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

class EditDistance:
    """
    Calculate edit distances between two strings.
    """
    def levenshtein_distance(self, s1, s2) -> int:
        """Levenshtein distance calculation."""
        # Check for equality and handle the trivial case where both strings are identical.
        if s1 == s2: return 0

        # Ensure s1 is the shorter string to optimize memory usage in the dynamic programming table.
        if len(s1) > len(s2): s1, s2 = s2, s1

        # Initialize the previous row of the dynamic programming table. This represents the number of edits
        # needed to transform an empty string into the first i characters of s1.
        prev_row = np.arange(len(s1) + 1)

        # Iterate over each character in the second string.
        for c2 in s2:
            # Store the top-left value (from the previous iteration) and increment the first cell
            # which represents transforming s1 into the first character of the current substring of s2.
            old_value, prev_row[0] = prev_row[0], prev_row[0] + 1

            # Iterate over each character in the first string.
            for i1, c1 in enumerate(s1):
                # Calculate the cost for substitution. It's 0 if characters are the same, else 1.
                # Compare this with the costs for insertion and deletion, and pick the minimum.
                # old_value represents the substitution cost (top-left cell).
                # prev_row[i1] + 1 represents the deletion cost (top cell).
                # prev_row[i1 + 1] + 1 represents the insertion cost (left cell).
                new_value = min(old_value + (c1 != c2), prev_row[i1] + 1, prev_row[i1 + 1] + 1)
                
                # Update old_value for the next iteration and set the calculated minimum edit distance
                # for the current cell.
                old_value, prev_row[i1 + 1] = prev_row[i1 + 1], new_value

        # After completing the iterations, the last element of prev_row contains the Levenshtein distance.
        return prev_row[-1]

    def damerau_levenshtein_distance(self, s1, s2) -> int:
        """Damerau-Levenshtein distance calculation."""
        if s1 == s2: return 0
        if len(s1) > len(s2): s1, s2 = s2, s1

        prev_prev_row, prev_row = np.zeros(len(s2) + 1, dtype=int), np.arange(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = np.zeros(len(s2) + 1, dtype=int)
            current_row[0] = i + 1
            for j, c2 in enumerate(s2):
                insertions = current_row[j] + 1
                deletions = prev_row[j + 1] + 1
                substitutions = prev_row[j] + (c1 != c2)
                cost = min(insertions, deletions, substitutions)

                if i > 0 and j > 0 and c1 == s2[j - 1] and s1[i - 1] == c2:
                    transpositions = prev_prev_row[j - 1] + 1
                    cost = min(cost, transpositions)

                current_row[j + 1] = cost

            prev_prev_row, prev_row = prev_row, current_row

        return prev_row[-1]

    def hamming_distance(self, s1, s2) -> int:
        """Hamming distance calculation."""
        if not isinstance(s1, str) or not isinstance(s2, str):
            raise TypeError("Inputs must be strings")
        if len(s1) != len(s2):
            raise ValueError(f"Hamming distance requires equal length strings. Lengths provided: {len(s1)} and {len(s2)}")
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def jaro_distance(self, s1, s2) -> float:
        """Jaro distance calculation."""
        if s1 == s2: return 1.0  # Quick return for identical strings
        if not s1 or not s2: return 0.0  # Handle empty strings

        len_s1, len_s2 = len(s1), len(s2)
        match_distance = (max(len_s1, len_s2) // 2) - 1

        s1_matches = np.zeros(len_s1, dtype=bool)
        s2_matches = np.zeros(len_s2, dtype=bool)
        matches = 0

        for i, c1 in enumerate(s1):
            start, end = max(0, i - match_distance), min(i + match_distance + 1, len_s2)
            for j, c2 in enumerate(s2[start:end], start):
                if not s1_matches[i] and not s2_matches[j] and c1 == c2:
                    s1_matches[i] = s2_matches[j] = True
                    matches += 1
                    break

        if not matches: return 0.0

        transpositions = sum(s1[i] != s2[j] for i, j in zip(np.where(s1_matches)[0], np.where(s2_matches)[0]))
        return ((matches / len_s1) + (matches / len_s2) + ((matches - transpositions / 2) / matches)) / 3.0

    def jaro_winkler_distance(self, s1, s2, p=0.1, max_prefix=4) -> float:
        """Jaro-Winkler distance calculation."""
        jaro_dist = self.jaro_distance(s1, s2)
        if jaro_dist < 0.7: return jaro_dist  # Threshold check
        prefix = sum(c1 == c2 for c1, c2 in list(zip(s1, s2))[:max_prefix])
        return jaro_dist + prefix * p * (1 - jaro_dist)

    def _initialize_dp_table(self, m, n) -> list:
        """Initialize the dynamic programming table for LCS computation."""
        return [[0] * (n + 1) for _ in range(m + 1)]

    def _calculate_lcs_length(self, X, Y, dp):
        """Calculate length of LCS using dynamic programming."""
        m, n = len(X), len(Y)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    def _find_all_lcs_sequences(self, X, Y, m, n, dp, memo) -> set:
        """Find all LCS sequences."""
        if m == 0 or n == 0:
            return set([""])
        if (m, n) in memo:
            return memo[(m, n)]

        sequences = set()
        if X[m - 1] == Y[n - 1]:
            lcs = self._find_all_lcs_sequences(X, Y, m - 1, n - 1, dp, memo)
            sequences = {Z + X[m - 1] for Z in lcs}
        else:
            if dp[m - 1][n] >= dp[m][n - 1]:
                sequences.update(self._find_all_lcs_sequences(X, Y, m - 1, n, dp, memo))
            if dp[m][n - 1] >= dp[m - 1][n]:
                sequences.update(self._find_all_lcs_sequences(X, Y, m, n - 1, dp, memo))

        memo[(m, n)] = sequences
        return sequences

    def longest_common_subsequence(self, s1, s2) -> int:
        """Calculates the length of the longest common subsequence between two strings."""
        if s1 == s2:
            return len(s1)  # Quick return for identical strings
        dp = self._initialize_dp_table(len(s1), len(s2))
        self._calculate_lcs_length(s1, s2, dp)
        return dp[-1][-1]

    def get_lcs_details(self, s1, s2) -> tuple:
        """Retrieves all LCS sequences and determines their contiguity."""
        dp = self._initialize_dp_table(len(s1), len(s2))
        self._calculate_lcs_length(s1, s2, dp)
        memo = {}
        lcs_sequences = self._find_all_lcs_sequences(s1, s2, len(s1), len(s2), dp, memo)
        is_contiguous = any(seq in s1 or seq in s2 for seq in lcs_sequences)
        return lcs_sequences, is_contiguous