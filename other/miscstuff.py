import nltk
import re
from string import ascii_lowercase
from collections import Counter
from typing import Set, Dict
from pathlib import Path

# Ensure NLTK corpora are downloaded
def download_corpus(corpus_name: str):
    """Download the specified NLTK corpus if not already present."""
    try:
        nltk.data.find(corpus_name)
    except LookupError:
        nltk.download(corpus_name, quiet=True)

def process_words(corpus_words: Set[str]) -> Set[str]:
    """Remove punctuation/non-alphanumeric characters and numbers."""
    return {re.sub(r'[^a-zA-Z]', '', word).lower() for word in corpus_words if word.isalpha()}

def calculate_letter_frequencies(words: Set[str], unique: bool = False) -> Dict[str, int]:
    """Calculate letter frequencies in the given set of words."""
    if unique:
        # Count type frequencies (each letter once per word)
        return Counter(letter for word in words for letter in set(word.lower()) if letter in ascii_lowercase)
    else:
        # Count token frequencies
        return Counter(letter for word in words for letter in word.lower() if letter in ascii_lowercase)

def calculate_percentages(frequencies: Dict[str, int], total: int) -> Dict[str, float]:
    """Calculate percentage frequencies."""
    return {letter: (count / total) * 100 for letter, count in frequencies.items()}

def print_frequencies(title: str, frequencies: Dict[str, float]):
    """Print sorted letter frequencies."""
    print(f"{title}:")
    for letter, freq in sorted(frequencies.items(), key=lambda item: item[1], reverse=True):
        print(f"{letter}: {freq:.2f}%")

def read_file_words(file_path: Path) -> Set[str]:
    """Read and process words from a file."""
    with file_path.open('r') as file:
        return process_words(set(file.read().split()))

def main():
    download_corpus('corpora/brown')
    download_corpus('corpora/cmudict')

    # Process Brown corpus
    brown_words = process_words(nltk.corpus.brown.words())
    token_freq = calculate_letter_frequencies(brown_words)
    type_freq = calculate_letter_frequencies(brown_words, unique=True)

    # Calculate and print token and type percentages
    total_tokens = sum(token_freq.values())
    total_types = sum(type_freq.values())

    token_percentage = calculate_percentages(token_freq, total_tokens)
    type_percentage = calculate_percentages(type_freq, total_types)

    print_frequencies("Token Letter Percentages", token_percentage)
    print("\nType Letter Percentages:")
    print_frequencies("Type Letter Percentages", type_percentage)

    # Process and count words from other sources
    for corpus_name, corpus_function in [("Brown", nltk.corpus.brown.words), 
                                         ("CMU", lambda: (word.lower() for word, _ in nltk.corpus.cmudict.entries()))]:
        words = process_words(set(corpus_function()))
        print(f"There are {len(words)} unique words in the {corpus_name} corpus.")

    # Read and process CLMET3.txt
    clmet3_words = read_file_words(Path('CLMET3.txt'))
    print(f"There are {len(clmet3_words)} unique words in the CLMET3 corpus.")

if __name__ == "__main__":
    main()