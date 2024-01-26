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
    tokenized_corpus = tokenizer.tokenize(' '.join(corpus_tokens), lowercase=True)

    # Perform basic and advanced analysis
    basic_analyzer = BasicCorpusAnalyzer(tokenized_corpus, shuffle_tokens=shuffle_tokens)
    advanced_analyzer = AdvancedCorpusAnalyzer(tokenized_corpus)
    hapax_legomena_count = sum(1 for _, details in basic_analyzer.token_details.items() if details['frequency'] == 1)

    # Log the results with improved formatting
    logger.info(f"Total Token Count: {basic_analyzer.total_token_count}")
    logger.info(f"Total Word Types (Distinct Tokens): {len(basic_analyzer.frequency)}")
    logger.info(f"Total Hapax Legomena (Unique Tokens): {hapax_legomena_count}")
    logger.info(f"Yule's K: {advanced_analyzer.yules_k():.6f}")
    logger.info(f"Herdan's C: {advanced_analyzer.herdans_c():.6f}")

    # Calculate and log alpha for Zipf's law
    alpha = advanced_analyzer.calculate_alpha()
    logger.info(f"Zipf Alpha: {alpha:.6f}")

    # Plot Zipf's law fit
    advanced_analyzer.plot_zipfs_law_fit(corpus_name)  # Adjusted to use only one argument

    K_sampling, Beta_sampling = advanced_analyzer.calculate_heaps_law_sampling()
    logger.info(f"Heap's K and Beta from Sampling: {K_sampling:.6f}, {Beta_sampling:.6f}")

    estimated_vocabulary_size = K_sampling * (basic_analyzer.total_token_count ** Beta_sampling)
    logger.info(f"Estimated Vocabulary Size (V) using Heaps' Law: {estimated_vocabulary_size:.2f}")

    # Fit Zipf-Mandelbrot distribution and log s and q parameters
    try:
        q, s = advanced_analyzer.fit_zipf_mandelbrot()
        logger.info(f"Zipf-Mandelbrot Parameters: s = {s:.6f}, q = {q:.6f}")

        # Plot the Zipf-Mandelbrot distribution fit
        advanced_analyzer.plot_zipf_mandelbrot_fit(q, s, corpus_name)
    except Exception as e:
        logger.error(f"Error in fitting Zipf-Mandelbrot distribution: {e}")

# Analyze multiple corpora
corpora = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']
for corpus in corpora:
    analyze_corpus(corpus)

