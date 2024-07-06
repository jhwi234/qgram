import nltk
from nltk.corpus import brown
from collections import Counter
import re
from pathlib import Path

def download_corpus():
    try:
        nltk.download('brown')
        print("Brown corpus downloaded successfully.")
    except Exception as e:
        print(f"An error occurred while downloading the Brown corpus: {e}")

def get_words_from_corpus():
    try:
        words = set(brown.words())
        return words
    except Exception as e:
        print(f"An error occurred while fetching words from the Brown corpus: {e}")
        return set()

def preprocess_word(word):
    """Remove non-alphabetic characters from a word and convert to lowercase."""
    word = word.lower()  # Convert to lowercase
    return re.sub('[^a-z]', '', word)  # Remove non-alphabetic characters

def generate_ngrams(word, n):
    """Generate character-level n-grams for a given word."""
    return [word[i:i+n] for i in range(len(word)-n+1)]

def count_ngrams(words, n_min=1, n_max=7):
    n_grams_counter = Counter()
    for word in words:
        clean_word = preprocess_word(word)
        if clean_word:  # Only process words that are not empty after cleaning
            for n in range(n_min, n_max + 1):
                n_grams_counter.update(generate_ngrams(clean_word, n))
    return n_grams_counter

def save_ngram_counts_to_file(n_grams_counter, output_file):
    try:
        # Sort n-grams by frequency in descending order
        sorted_ngrams = sorted(n_grams_counter.items(), key=lambda item: item[1], reverse=True)
        with output_file.open('w') as f:
            for ngram, freq in sorted_ngrams:
                f.write(f"{ngram}\t{freq}\n")
        print(f"Character-level n-grams frequency counts have been saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while saving n-grams to file: {e}")

def main():
    # Step 1: Download the Brown corpus
    download_corpus()

    # Step 2: Extract words from the Brown corpus
    words = get_words_from_corpus()

    # Step 3: Generate and count character-level n-grams
    n_grams_counter = count_ngrams(words)

    # Step 4: Save the frequency counts to a file
    output_file = Path("character_ngrams_frequency.txt")
    save_ngram_counts_to_file(n_grams_counter, output_file)

if __name__ == "__main__":
    main()
