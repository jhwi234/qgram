import logging
import cProfile
import pstats
import io
from corpus_analysis import CorpusTools, AdvancedTools, CorpusPlots, Tokenizer, CorpusLoader, EntropyCalculator

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def analyze_corpus(corpus_name, plots_to_generate=None, enable_profiling=False, entropy_analysis=False):
    logger.info(f"\nAnalyzing {corpus_name} Corpus\n")

    # Profiler setup
    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    # Load and tokenize the corpus
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True)
    tokenized_corpus = tokenizer.tokenize(corpus_tokens, lowercase=True)

    # Analysis based on the flags
    basic_analyzer = CorpusTools(tokenized_corpus, shuffle_tokens=False)
    results = [
        f"Token Count: {basic_analyzer.total_token_count}",
        f"Vocab Count: {len(basic_analyzer.frequency)}",
        f"Hapax Count: {len(basic_analyzer.x_legomena(1))}"
    ]

    if entropy_analysis:
        entropy_calculator = EntropyCalculator(tokenized_corpus)
        results.extend(perform_entropy_analysis(entropy_calculator))
    else:
        advanced_analyzer = AdvancedTools(tokenized_corpus)
        plotter = CorpusPlots(advanced_analyzer, corpus_name)
        results.extend(perform_advanced_analysis(advanced_analyzer, plotter, plots_to_generate))

    # Profiling and logging
    if enable_profiling and profiler is not None:
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('time', 'cumulative').print_stats(10)  # Focus on the top 10 by time and cumulative time
        logger.info("Profiling Results (Top 10 by Time and Cumulative Time):\n" + s.getvalue())

    for result in results:
        logger.info(result)

def perform_entropy_analysis(entropy_calculator):
    H0 = entropy_calculator.calculate_H0()
    H1 = entropy_calculator.calculate_H1()
    H3 = entropy_calculator.calculate_H3_kenlm()
    redundancy = entropy_calculator.calculate_redundancy(H3, H0)
    return [
        f"Zero-order Entropy (H0): {H0:.2f}",
        f"First-order Entropy (H1): {H1:.2f}",
        f"Third-order Entropy (H3): {H3:.2f}",
        f"Redundancy: {redundancy:.2f}%"
    ]

def perform_advanced_analysis(advanced_analyzer, plotter, plots_to_generate):
    results = []
    if "zipf" in plots_to_generate:
        alpha = advanced_analyzer.calculate_zipf_alpha()
        plotter.plot_zipfs_law_fit()
        results.append(f"Zipf Alpha: {alpha:.2f}")
    if "heaps" in plots_to_generate:
        K, beta = advanced_analyzer.calculate_heaps_law()
        results.append(f"Heaps K: {K:.2f}, Beta: {beta:.2f}")
    if "zipf_mandelbrot" in plots_to_generate:
        q, s = advanced_analyzer.calculate_zipf_mandelbrot()
        results.append(f"Zipf-Mandelbrot q: {q:.2f}, s: {s:.2f}")
    return results

# Example usage
corpora = ['brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'shakespeare', 'state_union', 'gutenberg']
for corpus in corpora:
    analyze_corpus(corpus, plots_to_generate=["zipf", "heaps", "zipf_mandelbrot"], enable_profiling=False, entropy_analysis=True)
