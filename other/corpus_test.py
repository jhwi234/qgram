import logging
from corpus_analysis import BasicCorpusAnalyzer, AdvancedCorpusAnalyzer, Tokenizer, CorpusLoader

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def analyze_corpus(corpus_name, shuffle_tokens=False):
    logger.info(f"\nAnalyzing {corpus_name} Corpus")

    # Load and tokenize the corpus
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True)
    tokenized_corpus = tokenizer.tokenize(corpus_tokens, lowercase=True)

    # Perform basic and advanced analysis
    basic_analyzer = BasicCorpusAnalyzer(tokenized_corpus, shuffle_tokens=shuffle_tokens)
    advanced_analyzer = AdvancedCorpusAnalyzer(tokenized_corpus)
    hapax_legomena_count = sum(1 for _, details in basic_analyzer.token_details.items() if details['frequency'] == 1)

    # Log basic analysis results
    logger.info(f"Total Token Count: {basic_analyzer.total_token_count}")
    logger.info(f"Total Word Types (Distinct Tokens): {len(basic_analyzer.frequency)}")
    logger.info(f"Total Hapax Legomena (Unique Tokens): {hapax_legomena_count}")

    # Advanced analysis - Yule's K and Herdan's C
    logger.info(f"Yule's K: {advanced_analyzer.yules_k():.6f}")
    logger.info(f"Herdan's C: {advanced_analyzer.herdans_c():.6f}")

    # Zipf's Law Analysis
    alpha = advanced_analyzer.calculate_alpha()
    logger.info(f"Zipf Alpha: {alpha:.6f}")
    advanced_analyzer.plot_zipfs_law_fit(corpus_name)

    # Heap's Law Analysis
    K, Beta = advanced_analyzer.calculate_heaps_law()
    logger.info(f"Heap's K and Beta: {K:.6f}, {Beta:.6f}")
    estimated_vocabulary_size = K * (basic_analyzer.total_token_count ** Beta)
    logger.info(f"Estimated Vocabulary Size (V) using Heaps' Law: {estimated_vocabulary_size:.2f}")
    advanced_analyzer.plot_heaps_law(K, Beta, corpus_name)

    # Zipf-Mandelbrot Distribution Analysis
    try:
        q, s = advanced_analyzer.fit_zipf_mandelbrot()
        logger.info(f"Zipf-Mandelbrot Parameters: s = {s:.6f}, q = {q:.6f}")
        advanced_analyzer.plot_zipf_mandelbrot_fit(q, s, corpus_name)
    except Exception as e:
        logger.error(f"Error in fitting Zipf-Mandelbrot distribution: {e}")

# Analyze multiple corpora
corpora = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']
for corpus in corpora:
    analyze_corpus(corpus)



