import nltk
from collections import Counter

def calculate_zipfian_probabilities(corpus_name):
    nltk.download(corpus_name)
    corpus = getattr(nltk.corpus, corpus_name).words()
    words = [word.lower() for word in corpus if word.isalpha()]
    word_freq = Counter(words)
    total_words = sum(word_freq.values())
    unique_word_types = len(word_freq)

    sorted_freqs = sorted(word_freq.values(), reverse=True)
    cumulative_freq = 0
    half_cumulative_freq = total_words / 2
    top_half_count = 0

    for freq in sorted_freqs:
        cumulative_freq += freq
        top_half_count += 1
        if cumulative_freq >= half_cumulative_freq:
            break

    prob_top_half = cumulative_freq / total_words
    prob_bottom_half = (total_words - cumulative_freq) / total_words

    return prob_top_half, prob_bottom_half, top_half_count, total_words, unique_word_types

corpora = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']

for corpus_name in corpora:
    prob_top_half, prob_bottom_half, top_half_count, total_words, unique_word_types = calculate_zipfian_probabilities(corpus_name)
    print(f"Corpus: {corpus_name}")
    print(f"Probability of a word being in the top half of token occurences: {prob_top_half:.4f}")
    print(f"Probability of a word being in the bottom half of token occurences: {prob_bottom_half:.4f}")
    print(f"Number of word types in the top half: {top_half_count}")
    print(f"Total number of word tokens in the corpus: {total_words}")
    print(f"Number of unique word types in the corpus: {unique_word_types}\n")
