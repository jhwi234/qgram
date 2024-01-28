import logging
from corpus_analysis import BasicCorpusAnalyzer, AdvancedCorpusAnalyzer, CorpusPlotter, Tokenizer, CorpusLoader

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
    plotter = CorpusPlotter(advanced_analyzer, corpus_name, plots_dir='plots')
    hapax_legomena_count = sum(1 for _, details in basic_analyzer.token_details.items() if details['frequency'] == 1)

    # Calculate and gather analysis results
    results = [
        f"Token Count: {basic_analyzer.total_token_count}",
        f"Types Count (Distinct Tokens): {len(basic_analyzer.frequency)}",
        f"Hapax Count (Unique Tokens): {hapax_legomena_count}",
        f"Yule's K: {advanced_analyzer.yules_k():.6f}",
        f"Herdan's C: {advanced_analyzer.herdans_c():.6f}",
        f"Zipf Alpha: {advanced_analyzer.calculate_alpha():.6f}"
    ]

    K, Beta = advanced_analyzer.calculate_heaps_law()
    estimated_vocabulary_size = advanced_analyzer.estimate_vocabulary_size(basic_analyzer.total_token_count)
    mean_deviation, std_deviation = advanced_analyzer.assess_alpha_fit(advanced_analyzer.calculate_alpha())

    results.extend([
        f"Alpha Fit Mean Deviation: {mean_deviation:.6f}", 
        f"ALpha Fit Standard Deviation: {std_deviation:.6f}",
        f"Heaps' K: {K:.6f} and Beta: {Beta:.6f}",
        f"Estimated Vocabulary Size: {estimated_vocabulary_size:.2f}",
        f"Actual Vocabulary Size: {len(basic_analyzer.frequency)}",
    ])

    # Calculate and assess Zipf-Mandelbrot parameters
    try:
        q, s = advanced_analyzer.calculate_zipf_mandelbrot()
        zipf_mandelbrot_mean_deviation, zipf_mandelbrot_std_deviation = advanced_analyzer.assess_zipf_mandelbrot_fit(q, s)
        results.append(f"Zipf-Mandelbrot params: s = {s:.6f}, q = {q:.6f}")
        results.append(f"Zipf-Mandelbrot Fit Mean Deviation: {zipf_mandelbrot_mean_deviation:.6f}")
        results.append(f"Zipf-Mandelbrot Fit Standard Deviation: {zipf_mandelbrot_std_deviation:.6f}")
    except Exception as e:
        results.append(f"Error in fitting Zipf-Mandelbrot distribution: {e}")

    # Log results
    for result in results:
        logger.info(result)

    # Generate plots using the plotter
    plotter.plot_zipfs_law_fit()
    plotter.plot_heaps_law(K, Beta)
    if 'Error' not in results[-1]:
        plotter.plot_zipf_mandelbrot_fit(q, s)

# Analyze multiple corpora
corpora = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'shakespeare', 'state_union']
for corpus in corpora:
    analyze_corpus(corpus)