import logging
from corpus_analysis import BasicCorpusAnalyzer, AdvancedCorpusAnalyzer, ZipfianAnalysis, Tokenizer, CorpusLoader

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def analyze_corpus(corpus_name, shuffle_tokens=False):
    logger.info(f"\nAnalyzing {corpus_name} Corpus\n")

    # Load and tokenize the corpus
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True)
    tokenized_corpus = tokenizer.tokenize(corpus_tokens, lowercase=True)

    # Perform basic and advanced analysis
    basic_analyzer = BasicCorpusAnalyzer(tokenized_corpus, shuffle_tokens=shuffle_tokens)
    advanced_analyzer = AdvancedCorpusAnalyzer(tokenized_corpus)
    zipfian_analyzer = ZipfianAnalysis(tokenized_corpus)
    hapax_legomena_count = sum(1 for _, details in basic_analyzer.token_details.items() if details['frequency'] == 1)

    # Calculate and gather analysis results
    results = [
        f"Total Token Count: {basic_analyzer.total_token_count}",
        f"Total Word Types (Distinct Tokens): {len(basic_analyzer.frequency)}",
        f"Total Hapax Legomena (Unique Tokens): {hapax_legomena_count}",
        f"Yule's K: {advanced_analyzer.yules_k():.6f}",
        f"Herdan's C: {advanced_analyzer.herdans_c():.6f}",
        f"Zipf Alpha: {advanced_analyzer.calculate_alpha():.6f}"
    ]

    K, Beta = advanced_analyzer.calculate_heaps_law()
    estimated_vocabulary_size = advanced_analyzer.estimate_vocabulary_size(basic_analyzer.total_token_count)
    mean_deviation, std_deviation = zipfian_analyzer.assess_zipfian_fit(advanced_analyzer.calculate_alpha())

    results.extend([
        f"Zipf Fit - Mean Deviation: {mean_deviation:.6f}", 
        f"Zipf Fit - Standard Deviation: {std_deviation:.6f}",
        f"Heap's K: {K:.6f} and Beta: {Beta:.6f}",
        f"Estimated Vocabulary Size using Heaps' Law: {estimated_vocabulary_size:.2f}"
    ])

    try:
        q, s = advanced_analyzer.fit_zipf_mandelbrot()
        results.append(f"Zipf-Mandelbrot params: s = {s:.6f}, q = {q:.6f}")
    except Exception as e:
        results.append(f"Error in fitting Zipf-Mandelbrot distribution: {e}")

    # Log results
    for result in results:
        logger.info(result)

    # Generate plots
    advanced_analyzer.plot_zipfs_law_fit(corpus_name)
    advanced_analyzer.plot_heaps_law(K, Beta, corpus_name)
    if 'Error' not in results[-1]:
        advanced_analyzer.plot_zipf_mandelbrot_fit(q, s, corpus_name)
    zipfian_analyzer.plot_zipfs_law_fit(corpus_name)

# Analyze multiple corpora
corpora = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'shakespeare', 'state_union']
for corpus in corpora:
    analyze_corpus(corpus)
