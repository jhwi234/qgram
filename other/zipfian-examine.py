import nltk
from collections import Counter

# Function to calculate probabilities based on Zipf's law for a given text corpus
def calculate_zipfian_probabilities(corpus_name):
    # Download the specified NLTK corpus, if not already present
    nltk.download(corpus_name, quiet=True)
    
    # Retrieve words from the corpus and clean them
    # Cleaning involves converting words to lowercase and filtering out non-alphabetic strings
    corpus_words = getattr(nltk.corpus, corpus_name).words()
    cleaned_words = [word.lower() for word in corpus_words if word.isalpha()]

    # Count the frequency of each word in the cleaned list
    # Counter creates a dictionary with words as keys and their counts as values
    word_frequencies = Counter(cleaned_words)
    
    # Calculate the total number of word occurrences in the corpus
    total_word_count = sum(word_frequencies.values())

    # Sort the word frequencies in descending order (most frequent word first)
    frequencies_sorted = sorted(word_frequencies.values(), reverse=True)

    # Initialize variables to calculate the cumulative sum of word frequencies
    # and to count the number of word types in the most frequent half
    cumulative_sum = 0
    top_half_word_types = 0

    # Loop through the sorted frequencies and accumulate their count
    # Stop accumulating once the sum reaches half of the total word count
    for frequency in frequencies_sorted:
        cumulative_sum += frequency
        top_half_word_types += 1
        if cumulative_sum >= total_word_count / 2:
            break

    # Calculate the probability of encountering a word from the most frequent half
    probability_top_half = cumulative_sum / total_word_count
    # Calculate the probability of encountering a word from the least frequent half
    probability_bottom_half = 1 - probability_top_half

    # Calculate the number of word types in the least frequent half
    bottom_half_word_types = len(word_frequencies) - top_half_word_types

    # Return the calculated probabilities, word type counts, and total counts
    return probability_top_half, probability_bottom_half, top_half_word_types, bottom_half_word_types, total_word_count, len(word_frequencies)

# Function to format the output in a readable way
def format_output(corpus, top_half_prob, bottom_half_prob, top_half_types, bottom_half_types, total_tokens, unique_types):
    # Construct a multi-line string with formatted output
    output = (
        f"Corpus: {corpus}\n"
        f"--------------------------------------------\n"
        f"Most Frequent Half - Word Probability   : {top_half_prob:.4f}\n"
        f"Least Frequent Half - Word Probability  : {bottom_half_prob:.4f}\n"
        f"Word Types in Most Frequent Half        : {top_half_types}\n"
        f"Word Types in Least Frequent Half       : {bottom_half_types}\n"
        f"Total Word Tokens                       : {total_tokens}\n"
        f"Total Word Types                        : {unique_types}\n"
    )
    return output

# List of corpora to analyze using the calculate_zipfian_probabilities function
corpora_to_analyze = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']

# Iterate over each corpus in the list
for corpus in corpora_to_analyze:
    # Calculate probabilities and counts for the current corpus
    top_half_prob, bottom_half_prob, top_half_types, bottom_half_types, total_tokens, unique_types = calculate_zipfian_probabilities(corpus)
    
    # Print the formatted results
    print(format_output(corpus, top_half_prob, bottom_half_prob, top_half_types, bottom_half_types, total_tokens, unique_types))
