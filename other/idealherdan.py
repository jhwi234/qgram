import numpy as np
import math

def _calculate_generalized_harmonic(n, alpha):
    """
    Calculate the generalized harmonic number for a given 'n' and 'alpha'.
    Used in simulating the Zipfian distribution.
    """
    # Sum the inverse of each rank (i) raised to the power of 'alpha', from 1 to 'n'.
    return np.sum(1 / np.power(np.arange(1, n + 1), alpha))

def find_vocabulary_size(corpus_token_size, alpha=1):
    """
    Find the vocabulary size for a given corpus token size and alpha,
    assuming a Zipfian distribution.
    """
    # Initialize the search range for vocabulary size
    low, high = 1, corpus_token_size

    while low < high:
        mid = (low + high) // 2
        harmonic_number = _calculate_generalized_harmonic(mid, alpha)

        # Estimate the total frequency for the mid vocabulary size
        estimated_total_freq = sum((corpus_token_size / harmonic_number) / (rank ** alpha) for rank in range(1, mid + 1))

        # Adjust the search range based on the estimated total frequency
        if estimated_total_freq < corpus_token_size:
            low = mid + 1
        else:
            high = mid

    return low

def simulate_zipfian_corpus(corpus_token_size, alpha=1):
    """
    Simulate a corpus that follows a Zipfian distribution.
    """
    # Determine the vocabulary size for the corpus
    vocabulary_size = find_vocabulary_size(corpus_token_size, alpha)
    harmonic_number = _calculate_generalized_harmonic(vocabulary_size, alpha)

    # Generate the ranks and calculate the expected frequency of each word
    ranks = np.arange(1, vocabulary_size + 1)
    frequencies = (corpus_token_size / harmonic_number) / np.power(ranks, alpha)

    # Create unique word tokens and repeat each word by its calculated frequency
    words = np.array([f'word{i}' for i in ranks])
    corpus = np.repeat(words, np.round(frequencies).astype(int))

    # Shuffle the corpus to simulate natural word order
    np.random.shuffle(corpus)

    # Join words to form a single string, truncated to the specified corpus size
    return ' '.join(corpus[:corpus_token_size]), vocabulary_size

def calculate_herdans_c(corpus):
    """
    Calculate Herdan's C, a measure of lexical diversity.
    """
    # Split the corpus into tokens and count unique types
    tokens = corpus.split()
    types = len(set(tokens))

    # Avoid division by zero and calculate Herdan's C
    if types == 0 or len(tokens) == 0:
        return 0
    return math.log(types) / math.log(len(tokens))

# Example usage
corpus_token_size = 1000000
zipfian_corpus, vocab_size = simulate_zipfian_corpus(corpus_token_size)
herdans_c_value = calculate_herdans_c(zipfian_corpus)
print(f"Vocabulary Size: {vocab_size}, Herdan's C: {herdans_c_value}")
