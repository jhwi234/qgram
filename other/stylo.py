from collections import Counter
from scipy.stats import chisquare
import numpy as np
from corpus_analysis import CorpusTools, Tokenizer, CorpusLoader

def mendenhall_curves(tokens):
    """
    Generates Mendenhall's characteristic curves for a given text.
    Improved error handling and data validation.
    """
    if not tokens:
        raise ValueError("Token list is empty.")
    length_distribution = Counter(len(word) for word in tokens if word.isalpha())  # Consider only alphabetic words
    return length_distribution

def kilgariff_chi_squared(freq_dict1, freq_dict2, smoothing_factor=0.5):
    """
    Applies Kilgariff's Chi-Squared method with improved handling for zero frequencies.
    """
    if not freq_dict1 or not freq_dict2:
        raise ValueError("Frequency dictionaries are empty or not provided.")

    all_words = set(freq_dict1.keys()) | set(freq_dict2.keys())
    freqs1 = [freq_dict1.get(word, 0) + smoothing_factor for word in all_words]
    freqs2 = [freq_dict2.get(word, 0) + smoothing_factor for word in all_words]
    return chisquare(freqs1, f_exp=freqs2).statistic

def burrows_delta(frequency_distributions, smoothing_factor=0.5):
    """
    Computes Burrows' Delta with enhanced error checking and data validation.
    """
    if not frequency_distributions:
        raise ValueError("Frequency distributions list is empty.")

    all_words = set.union(*(set(dist.keys()) for dist in frequency_distributions))
    means = {word: np.mean([dist.get(word, 0) for dist in frequency_distributions]) for word in all_words}
    std_devs = {word: np.std([dist.get(word, 0) for dist in frequency_distributions], ddof=1) + smoothing_factor for word in all_words}
    z_scores = [[(dist.get(word, 0) - means[word]) / std_devs[word] for word in all_words] for dist in frequency_distributions]
    deltas = [np.mean([abs(score) for score in scores]) for scores in z_scores]
    return deltas

def main():
    try:
        # Initialization and data loading with error handling
        corpus_loader = CorpusLoader("brown")  # Replace with the actual corpus name
        tokenizer = Tokenizer(remove_punctuation=True, remove_stopwords=True, use_nltk_tokenizer=True)
        corpus_text = corpus_loader.load_corpus()
        tokens = tokenizer.tokenize(corpus_text)

        if not tokens:
            raise RuntimeError("Failed to tokenize the corpus.")

        # Corpus analysis
        analyzer = CorpusTools(tokens)
        mendenhall_result = mendenhall_curves(tokens)
        print("Mendenhall's Characteristic Curves:", mendenhall_result)

        # Example texts - replace with actual text samples
        text_sample1 = "your first text sample"
        text_sample2 = "your second text sample"
        tokens1 = tokenizer.tokenize(text_sample1)
        tokens2 = tokenizer.tokenize(text_sample2)
        freq_dict1 = Counter(tokens1)
        freq_dict2 = Counter(tokens2)

        chi_squared_result = kilgariff_chi_squared(freq_dict1, freq_dict2)
        print("Kilgariff’s Chi-Squared Result:", chi_squared_result)

        # Burrows’ Delta Method
        texts_tokens = [tokens1, tokens2]  # Replace with actual token lists
        freq_distributions = [Counter(text) for text in texts_tokens]
        delta_results = burrows_delta(freq_distributions)
        print("Burrows’ Delta Results:", delta_results)

    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")

if __name__ == "__main__":
    main()