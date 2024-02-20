# Import statements remain unchanged
import nltk
import re
from string import ascii_lowercase
from collections import Counter
from typing import List, Dict, Iterable
from pathlib import Path

def download_corpus(corpus_name: str):
    try:
        nltk.data.find(corpus_name)
    except LookupError:
        nltk.download(corpus_name, quiet=True)

def process_words(corpus_words: Iterable[str]) -> List[str]:
    return [re.sub(r'[^a-zA-Z]', '', word).lower() for word in corpus_words if word.isalpha()]

def calculate_top_n_words(words: Iterable[str], n: int) -> List[tuple]:
    return Counter(words).most_common(n)

def calculate_letter_frequencies(words: Iterable[str], unique: bool = False) -> Dict[str, int]:
    if unique:
        return Counter(letter for word in words for letter in set(word) if letter in ascii_lowercase)
    else:
        return Counter(letter for word in words for letter in word if letter in ascii_lowercase)

def calculate_percentages(frequencies: Dict[str, int], total: int) -> Dict[str, float]:
    return {letter: (count / total) * 100 for letter, count in frequencies.items()}

def print_frequencies(title: str, frequencies: Dict[str, float]):
    print(f"\n{title}:")
    print("-" * 30)
    print(f"{'Letter':<10}{'Frequency (%)':>10}")
    print("-" * 30)
    for letter, freq in sorted(frequencies.items(), key=lambda item: item[1], reverse=True):
        print(f"{letter:<10}{freq:>10.2f}")

def analyze_corpus(corpus_name: str, corpus_words: Iterable[str]):
    words = process_words(corpus_words)
    token_freq = calculate_letter_frequencies(words)
    type_freq = calculate_letter_frequencies(words, unique=True)

    total_tokens = sum(token_freq.values())
    total_types = sum(type_freq.values())

    token_percentage = calculate_percentages(token_freq, total_tokens)
    type_percentage = calculate_percentages(type_freq, total_types)

    print_frequencies(f"Token Letter Percentages in {corpus_name} Corpus", token_percentage)
    print_frequencies(f"Type Letter Percentages in {corpus_name} Corpus", type_percentage)

    print(f"\nUnique Word Count in {corpus_name} Corpus: {len(set(words))}")

    # Conditional check to only run this for the Brown corpus
    if corpus_name == "Brown":
        top_n_words = calculate_top_n_words(words, 10)
        print(f"\nTop 10 Words in {corpus_name} Corpus:")
        print(f"{'Word':<15}{'Frequency':>10}")
        for word, freq in top_n_words:
            print(f"{word:<15}{freq:>10}")

# Updated to return a list of words from a file
def read_file_words(file_path: Path) -> List[str]:
    with file_path.open('r') as file:
        return process_words(file.read().split())

# Main function remains unchanged
def main():
    print("Downloading and Processing Corpora")
    download_corpus('corpora/brown')
    download_corpus('corpora/cmudict')

    analyze_corpus("Brown", nltk.corpus.brown.words())
    analyze_corpus("CMU", (word.lower() for word, _ in nltk.corpus.cmudict.entries()))
    analyze_corpus("CLMET3", read_file_words(Path('data/corpora/CLMET3.txt')))

if __name__ == "__main__":
    main()
