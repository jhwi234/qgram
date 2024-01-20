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

# Step 1: Load the Brown Corpus
corpus_loader = CorpusLoader('brown')
brown_tokens = corpus_loader.load_corpus()

# Step 2: Tokenize the Corpus
tokenizer = Tokenizer(remove_punctuation=True)
tokenized_brown = tokenizer.tokenize(brown_tokens)

# Step 3: Basic Analysis
basic_analyzer = BasicCorpusAnalyzer(tokenized_brown)
print("Median Token:", basic_analyzer.find_median_token())
print("Mean Token Frequency:", basic_analyzer.mean_token_frequency())
print("Query 25th most frequent token:", basic_analyzer.query_by_rank(25))
print("Query frequency and rank of 'university':", basic_analyzer.query_by_token('university'))
# Step 4: Advanced Analysis
advanced_analyzer = AdvancedCorpusAnalyzer(tokenized_brown)
# Display words in a certain rank range
print("Words in rank range 100 to 110:", advanced_analyzer.list_tokens_in_rank_range(100, 110))

print("\nCumulative Frequency Analysis (Top 20%):", advanced_analyzer.cumulative_frequency_analysis(0, 20))
print("Yule's K Measure:", advanced_analyzer.yules_k())
print("Herdan's C Measure:", advanced_analyzer.herdans_c())

# Step 5: Zipfian Analysis
zipfian_analyzer = ZipfianAnalysis(tokenized_brown)
zipfian_analyzer.plot_zipfian_comparison()
alpha = zipfian_analyzer.calculate_alpha()
print("\nEstimated Alpha for Zipfian Distribution:", alpha)
mean_deviation, std_deviation = zipfian_analyzer.assess_zipfian_fit(alpha)
print("Mean Deviation in Zipfian Fit:", mean_deviation)
print("Standard Deviation in Zipfian Fit:", std_deviation)
