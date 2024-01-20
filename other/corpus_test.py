import nltk
from corpus_analysis import BasicCorpusAnalyzer, ZipfianAnalysis, Tokenizer, CorpusLoader
import numpy as np

def download_nltk_corpora(corpus_names):
    """Download necessary NLTK corpora if not already available."""
    for corpus_name in corpus_names:
        try:
            nltk.data.find(f"corpora/{corpus_name}")
        except LookupError:
            print(f"Downloading NLTK corpus: {corpus_name}")
            nltk.download(corpus_name)

def cosine_similarity_manual(vec1, vec2):
    # Compute cosine similarity manually using numpy
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0

def lexical_similarity_analysis(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0  # Return zero similarity for empty token lists

    # Initialize analyzers
    analyzer1 = BasicCorpusAnalyzer(tokens1)
    analyzer2 = BasicCorpusAnalyzer(tokens2)

    # Overlap Coefficient
    common_vocab = set(analyzer1.frequency.keys()).intersection(analyzer2.frequency.keys())
    overlap_coefficient = len(common_vocab) / min(len(analyzer1.frequency), len(analyzer2.frequency))

    # Cosine Similarity
    all_vocab = set(analyzer1.frequency.keys()).union(analyzer2.frequency.keys())
    freq_vector1 = np.array([analyzer1.frequency.get(word, 0) for word in all_vocab])
    freq_vector2 = np.array([analyzer2.frequency.get(word, 0) for word in all_vocab])
    cos_similarity = cosine_similarity_manual(freq_vector1, freq_vector2)

    # Zipfian Comparison
    zipfian1 = ZipfianAnalysis(tokens1)
    zipfian2 = ZipfianAnalysis(tokens2)
    alpha1 = zipfian1.calculate_alpha()
    alpha2 = zipfian2.calculate_alpha()
    zipfian_similarity = 1 - abs(alpha1 - alpha2) / max(alpha1, alpha2)

    # Final Similarity Score (weighted average)
    final_score = (overlap_coefficient + cos_similarity + zipfian_similarity) / 3
    return final_score

def get_corpus_text(corpus_name):
    """Load and return the text of a given NLTK corpus."""
    try:
        corpus = getattr(nltk.corpus, corpus_name)
        return ' '.join(corpus.words()) if hasattr(corpus, 'words') else ''
    except LookupError as e:
        print(f"Error loading corpus {corpus_name}: {e}")
        return ''

def get_corpus_names():
    """Return a list of commonly used NLTK corpus names."""
    return [
        'brown',        # Brown Corpus
        'gutenberg',    # Project Gutenberg Selections
        'reuters',      # Reuters Corpus
        'inaugural',    # Inaugural Address Corpus
        'nps_chat',     # NPS Chat Corpus
        'webtext',      # Web and Chat Text
        'udhr'         # Universal Declaration of Human Rights
    ]

# Ensure NLTK corpora are downloaded
corpus_names = get_corpus_names()
download_nltk_corpora(corpus_names)

# Tokenize and Load texts from all available NLTK corpora
tokenizer = Tokenizer(remove_punctuation=True)
tokenized_corpora = {name: tokenizer.tokenize(get_corpus_text(name)) for name in corpus_names if get_corpus_text(name)}

# Compute the lexical similarity scores between all pairs of corpora
similarity_scores = {}
for i, corpus1 in enumerate(corpus_names):
    for corpus2 in corpus_names[i+1:]:
        if corpus1 in tokenized_corpora and corpus2 in tokenized_corpora:
            score = lexical_similarity_analysis(tokenized_corpora[corpus1], tokenized_corpora[corpus2])
            similarity_scores[(corpus1, corpus2)] = score

# Print similarity scores
for pair, score in similarity_scores.items():
    print(f"Lexical Similarity Score between {pair[0]} and {pair[1]}: {score}")