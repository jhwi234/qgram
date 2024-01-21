from corpus_analysis import BasicCorpusAnalyzer, AdvancedCorpusAnalyzer, ZipfianAnalysis, Tokenizer, CorpusLoader
import numpy as np

def cosine_similarity_manual(vec1, vec2):
    # Compute cosine similarity manually using numpy
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0

def lexical_similarity_analysis(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0  # Return zero similarity for empty token lists

    analyzer1 = BasicCorpusAnalyzer(tokens1)
    analyzer2 = BasicCorpusAnalyzer(tokens2)

    common_vocab = set(analyzer1.frequency.keys()).intersection(analyzer2.frequency.keys())
    overlap_coefficient = len(common_vocab) / min(len(analyzer1.frequency), len(analyzer2.frequency))

    all_vocab = set(analyzer1.frequency.keys()).union(analyzer2.frequency.keys())
    freq_vector1 = np.array([analyzer1.frequency.get(word, 0) for word in all_vocab])
    freq_vector2 = np.array([analyzer2.frequency.get(word, 0) for word in all_vocab])
    cos_similarity = cosine_similarity_manual(freq_vector1, freq_vector2)

    zipfian1 = ZipfianAnalysis(tokens1)
    zipfian2 = ZipfianAnalysis(tokens2)
    alpha1 = zipfian1.calculate_alpha()
    alpha2 = zipfian2.calculate_alpha()
    zipfian_similarity = 1 - abs(alpha1 - alpha2) / max(alpha1, alpha2)

    final_score = (overlap_coefficient + cos_similarity + zipfian_similarity) / 3
    return final_score

def analyze_corpus(corpus_name):
    print(f"\nAnalyzing {corpus_name} Corpus")

    # Load and tokenize the corpus
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True, use_nltk_tokenizer=True)
    tokenized_corpus = tokenizer.tokenize(corpus_tokens)

    # Perform basic and advanced analysis
    basic_analyzer = BasicCorpusAnalyzer(tokenized_corpus)
    advanced_analyzer = AdvancedCorpusAnalyzer(tokenized_corpus)
    print("Total Token Count:", basic_analyzer.total_token_count)
    print("Total Word Types (Distinct Tokens):", len(basic_analyzer.frequency))
    hapax_legomena_count = sum(1 for _, details in basic_analyzer.token_details.items() if details['frequency'] == 1)
    print(f"Total Hapax Legomena (Unique Tokens): {hapax_legomena_count}")
    print("Yule's K:", advanced_analyzer.yules_k())
    print("Herdan's C:", advanced_analyzer.herdans_c())
    print("Zipf Alpha:", advanced_analyzer.calculate_alpha())

    # Calculate K and Beta Measures for Heaps' Law and Zipf-Mandelbrot Parameters
    k_beta_measures = advanced_analyzer.calculate_heaps_law_constants()
    print("Heap's K and Beta:", k_beta_measures)

    # Estimated Vocabulary Size using Heaps' Law
    K, Beta = k_beta_measures
    N = basic_analyzer.total_token_count
    estimated_V = K * N**Beta
    actual_V = len(basic_analyzer.frequency)
    print("Estimated Vocabulary Size (V) using Heaps' Law:", estimated_V)
    print("Actual Vocabulary Size:", actual_V)

# Analyze multiple corpora
corpora = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']
for corpus in corpora:
    analyze_corpus(corpus)


