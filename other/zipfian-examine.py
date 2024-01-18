import nltk
from collections import Counter

def calculate_zipfian_probabilities(corpus_name):
    nltk.download(corpus_name)
    corpus_words = getattr(nltk.corpus, corpus_name).words()
    cleaned_words = [word.lower() for word in corpus_words if word.isalpha()]
    # Counts how many times each word type appears in the corpus
    word_frequencies = Counter(cleaned_words)
    # Sums up the total number of word tokens in the corpus
    total_word_count = sum(word_frequencies.values())
    # Counts the tokens of unique word types in the corpus
    distinct_word_count = len(word_frequencies)
    # Sorts the word frequencies in ascending order (from least frequent to most frequent)
    frequencies_sorted = sorted(word_frequencies.values())
    # Initializes variables for tracking the cumulative frequency
    cumulative_count = 0
    mid_word_count = total_word_count / 2
    top_half_word_types = 0
    # Iterates through the sorted frequencies to find out how many of the word types
    # are needed to reach half of the total word count
    for frequency in frequencies_sorted:
        cumulative_count += frequency  # Adds the current word type's frequency to the cumulative total
        top_half_word_types += 1       # Counts how many word types have been added

        # Checks if the cumulative count has reached half of the total word count
        if cumulative_count >= mid_word_count:
            break  # Stops the loop when half of the word count is reached
    # Calculates the probability of encountering a word type from the top half of most frequent word types
    probability_top_half = cumulative_count / total_word_count
    # Calculates the probability of encountering a word type from the bottom half of least frequent word types
    probability_bottom_half = (total_word_count - cumulative_count) / total_word_count
    # Returns the calculated probabilities and word token counts
    return probability_top_half, probability_bottom_half, top_half_word_types, total_word_count, distinct_word_count

corpora_to_analyze = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']

for corpus in corpora_to_analyze:
    top_half_prob, bottom_half_prob, top_half_types, total_tokens, unique_types = calculate_zipfian_probabilities(corpus)
    
    print(f"Corpus: {corpus}")
    print(f"Probability of encountering a word from the most frequent half: {top_half_prob:.4f}")
    print(f"Probability of encountering a word from the least frequent half: {bottom_half_prob:.4f}")
    print(f"Number of unique word types in the most frequent half: {top_half_types}")
    print(f"Total number of word tokens in the corpus: {total_tokens}")
    print(f"Total number of unique word types in the corpus: {unique_types}\n")
