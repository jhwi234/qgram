import numpy as np
import math

def _calculate_generalized_harmonic(n, alpha):
    # Calculate the generalized harmonic number for given 'n' and 'alpha'.
    # This is used in simulating the Zipfian distribution. It sums the
    # inverse of each rank (i) raised to the power of 'alpha', from 1 to 'n'.
    # Use NumPy for efficient vectorized calculation of the generalized harmonic number.
    return np.sum(1 / np.power(np.arange(1, n + 1), alpha))

def find_vocabulary_size(corpus_token_size, alpha=1):
    # Use binary search to efficiently find the vocabulary (word types) size that fits
    # the Zipfian distribution for a given corpus length. This is because
    # the Zipfian distribution's total frequency for a given vocabulary size
    # is not linear and requires iterative searching.
    low, high = 1, corpus_token_size
    while low < high:
        mid = (low + high) // 2
        harmonic_number = _calculate_generalized_harmonic(mid, alpha)
        # Calculate the total frequency across all ranks up to 'mid'
        # using the Zipfian distribution formula. This checks if the
        # sum of frequencies up to this point meets the corpus length.
        total_freq = sum((corpus_token_size / harmonic_number) / (rank ** alpha) for rank in range(1, mid + 1))
        # Binary search logic to converge on the correct vocabulary size.
        if total_freq < corpus_token_size:
            low = mid + 1
        else:
            high = mid
    return low

def simulate_zipfian_corpus(corpus_token_size, alpha=1):
    # Simulate a corpus that follows a Zipfian distribution.
    vocabulary_size = find_vocabulary_size(corpus_token_size, alpha)
    harmonic_number = _calculate_generalized_harmonic(vocabulary_size, alpha)
    # Generate ranks from 1 to the vocabulary size.
    ranks = np.arange(1, vocabulary_size + 1)
    # Calculate the expected frequency of each word in the distribution.
    frequencies = (corpus_token_size / harmonic_number) / np.power(ranks, alpha)
    # Create word tokens. Each word is unique and is labeled as 'wordi'.
    words = [f'word{i}' for i in ranks]
    # Repeat each word by its calculated frequency and round it to the nearest integer.
    corpus = np.repeat(words, np.round(frequencies).astype(int))
    # Shuffle the corpus to simulate a natural sequence of words.
    np.random.shuffle(corpus)
    # Join the words to form a single string, truncated to the corpus length.
    return ' '.join(corpus[:corpus_token_size]), vocabulary_size

def calculate_herdans_c(corpus):
    # Calculate Herdan's C, a measure of lexical diversity.
    tokens = corpus.split()
    types = len(set(tokens))
    # Prevent division by zero if there are no types or tokens.
    if types == 0 or len(tokens) == 0:
        return 0
    # Herdan's C is the log of the number of unique words (types)
    # divided by the log of the total number of words (tokens).
    return math.log(types) / math.log(len(tokens))

# Example usage
corpus_token_size = 1000000  # Specify the total number of word tokens in the corpus.

# Simulate a Zipfian corpus and calculate Herdan's C for it.
zipfian_corpus, vocab_size = simulate_zipfian_corpus(corpus_token_size)
herdans_c_value = calculate_herdans_c(zipfian_corpus)
print(f"Vocabulary Size: {vocab_size}, Herdan's C: {herdans_c_value}")
