from corpus_analysis import BasicCorpusAnalyzer
from corpus_analysis import ZipfianAnalysis
from corpus_analysis import Tokenizer
from corpus_analysis import CorpusLoader
import numpy as np

def cosine_similarity_manual(vec1, vec2):
    # Compute cosine similarity manually using numpy
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0

def lexical_similarity_analysis(corpus1, corpus2):
    # Tokenize and Analyze frequency distribution together
    tokenizer = Tokenizer(remove_punctuation=True)
    analyzer1 = BasicCorpusAnalyzer(tokenizer.tokenize(corpus1))
    analyzer2 = BasicCorpusAnalyzer(tokenizer.tokenize(corpus2))

    if not analyzer1.tokens or not analyzer2.tokens:
        return 0  # Return zero similarity for empty corpora

    # Overlap Coefficient
    common_vocab = set(analyzer1.frequency.keys()).intersection(analyzer2.frequency.keys())
    overlap_coefficient = len(common_vocab) / min(len(analyzer1.frequency), len(analyzer2.frequency))

    # Cosine Similarity
    freq_vector1 = np.array([analyzer1.frequency.get(word, 0) for word in common_vocab])
    freq_vector2 = np.array([analyzer2.frequency.get(word, 0) for word in common_vocab])
    cos_similarity = cosine_similarity_manual(freq_vector1, freq_vector2)

    # Zipfian Comparison
    zipfian1 = ZipfianAnalysis(tokenizer.tokenize(corpus1))
    zipfian2 = ZipfianAnalysis(tokenizer.tokenize(corpus2))
    alpha1 = zipfian1.calculate_alpha()
    alpha2 = zipfian2.calculate_alpha()
    zipfian_similarity = 1 - abs(alpha1 - alpha2) / max(alpha1, alpha2)

    # Final Similarity Score (weighted average)
    final_score = (overlap_coefficient + cos_similarity + zipfian_similarity) / 3
    return final_score

# Example usage
def get_corpus_text(corpus_name):
    corpus_loader = CorpusLoader(corpus_name)
    tokens = corpus_loader.load_corpus()
    return ' '.join(tokens)

# Load text from the Brown and Gutenberg corpora
corpus_brown = get_corpus_text('brown')
corpus_gutenberg = get_corpus_text('gutenberg')

# Compute the lexical similarity score
similarity_score = lexical_similarity_analysis(corpus_brown, corpus_gutenberg)
print(f"Lexical Similarity Score: {similarity_score}")
