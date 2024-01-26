from corpus_analysis import BasicCorpusAnalyzer, AdvancedCorpusAnalyzer, Tokenizer, CorpusLoader

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


