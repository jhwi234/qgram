import logging
import cProfile
import pstats
import io
from corpus_analysis import CorpusTools, AdvancedTools, CorpusPlots, Tokenizer, CorpusLoader

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def analyze_corpus(corpus_name, shuffle_tokens=False, plots_to_generate=None, enable_profiling=False):
    if plots_to_generate is None:
        plots_to_generate = ["zipf", "heaps", "zipf_mandelbrot"]

    logger.info(f"\nAnalyzing {corpus_name} Corpus\n")

    # Profiler setup
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    # Load and tokenize the corpus
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True)
    tokenized_corpus = tokenizer.tokenize(corpus_tokens, lowercase=True)

    # Perform basic and advanced analysis
    basic_analyzer = CorpusTools(tokenized_corpus, shuffle_tokens=shuffle_tokens)
    advanced_analyzer = AdvancedTools(tokenized_corpus)
    plotter = CorpusPlots(advanced_analyzer, corpus_name)
    hapax_legomena_count = sum(1 for _, details in basic_analyzer.token_details.items() if details['frequency'] == 1)

    # Gather basic analysis results
    results = [
        f"Token Count: {basic_analyzer.total_token_count}",
        f"Types Count (Distinct Tokens): {len(basic_analyzer.frequency)}",
        f"Hapax Count (Unique Tokens): {hapax_legomena_count}",
        f"Yule's K: {advanced_analyzer.yules_k():.6f}",
        f"Herdan's C: {advanced_analyzer.herdans_c():.6f}",
    ]

    # Perform conditional calculations and plot generations
    if "zipf" in plots_to_generate:
        alpha, c = advanced_analyzer.calculate_alpha()
        mean_deviation, std_deviation = advanced_analyzer.assess_alpha_fit()
        plotter.plot_zipfs_law_fit()
        results.extend([
            f"Zipf Alpha: {alpha:.6f}",
            f"Zipf C: {c:.6f}",
            f"Alpha Fit Mean Deviation: {mean_deviation:.6f}", 
            f"Alpha Fit Standard Deviation: {std_deviation:.6f}"
        ])

    if "heaps" in plots_to_generate:
        K, Beta = advanced_analyzer.calculate_heaps_law()
        estimated_vocabulary_size = advanced_analyzer.estimate_vocabulary_size(basic_analyzer.total_token_count)
        plotter.plot_heaps_law()
        results.extend([
            f"Heaps' K: {K:.6f} and Beta: {Beta:.6f}",
            f"Estimated Vocabulary Size: {estimated_vocabulary_size:.2f}",
            f"Actual Vocabulary Size: {len(basic_analyzer.frequency)}"
        ])

    if "zipf_mandelbrot" in plots_to_generate:
        try:
            q, s = advanced_analyzer.calculate_zipf_mandelbrot()
            zipf_mandelbrot_mean_deviation, zipf_mandelbrot_std_deviation = advanced_analyzer.assess_zipf_mandelbrot_fit(q, s)
            plotter.plot_zipf_mandelbrot_fit()
            results.extend([
                f"Zipf-Mandelbrot params: s = {s:.6f}, q = {q:.6f}",
                f"Zipf-Mandelbrot Fit Mean Deviation: {zipf_mandelbrot_mean_deviation:.6f}",
                f"Zipf-Mandelbrot Fit Standard Deviation: {zipf_mandelbrot_std_deviation:.6f}"
            ])
        except Exception as e:
            results.append(f"Error in fitting Zipf-Mandelbrot distribution: {e}")

    # Profiler teardown and log profiling results
    if enable_profiling:
        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats(sortby)
        ps.print_stats(10)
        profiling_results = s.getvalue().split('\n')
        logger.info("\nProfiling Results (Top 10 Time-Consuming Functions):\n")
        for line in profiling_results[:10]:
            logger.info(line)

    # Log analysis results
    for result in results:
        logger.info(result)

# Example usage
corpora = ['brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'shakespeare', 'state_union', 'gutenberg']
plots_required = ["zipf"]
for corpus in corpora:
    analyze_corpus(corpus, plots_to_generate=plots_required, enable_profiling=False)
