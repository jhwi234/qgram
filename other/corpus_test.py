import logging
from pathlib import Path

from corpus_analysis import AdvancedTools, CorpusLoader, CorpusPlots, Tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s')
logger = logging.getLogger(__name__)

def load_and_tokenize_corpus(corpus_name):
    logger.info(f"Loading and tokenizing {corpus_name}...")
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True, remove_stopwords=False)
    tokenized_corpus = tokenizer.tokenize(' '.join(corpus_tokens), lowercase=True)
    return tokenized_corpus

def perform_advanced_analysis(corpus_name, plots_to_generate):
    tokenized_corpus = load_and_tokenize_corpus(corpus_name)
    advanced_analyzer = AdvancedTools(tokenized_corpus)
    
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    plotter = CorpusPlots(advanced_analyzer, corpus_name, plots_dir=plots_dir)

    results = []
    for plot_type in plots_to_generate:
        if plot_type == "heaps":
            K, beta = advanced_analyzer.calculate_heaps_law()
            plotter.plot_heaps_law()
            estimated_vocab_size = advanced_analyzer.estimate_vocabulary_size(advanced_analyzer.total_token_count)
            actual_vocab_size = len(advanced_analyzer.vocabulary())
            results.append(f"Heaps' Law Parameters for {corpus_name}: K={K:.2f}, Beta={beta:.3f}")
            results.append(f"Estimated vocabulary size using Heaps' Law: {estimated_vocab_size}")
            results.append(f"Actual vocabulary size: {actual_vocab_size}")
            # Calculate and log the difference or percentage error between estimated and actual vocabulary sizes
            difference = abs(estimated_vocab_size - actual_vocab_size)
            percentage_error = (difference / actual_vocab_size) * 100
            results.append(f"Difference: {difference}, Percentage Error: {percentage_error:.2f}%")
    log_results(results)

def log_results(results):
    for result in results:
        logger.info(result)

# Example usage
corpora = ['brown', 'reuters', 'webtext']
plots_to_generate = ["zipf", "heaps", "zipf_mandelbrot"]

for corpus in corpora:
    perform_advanced_analysis(corpus, plots_to_generate)
