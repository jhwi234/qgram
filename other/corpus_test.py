import logging
from corpus_analysis import BasicCorpusAnalyzer, AdvancedCorpusAnalyzer, Tokenizer, CorpusLoader

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def analyze_corpus(corpus_name):
    def heaps_law_analysis(shuffle_tokens):
        basic_analyzer = BasicCorpusAnalyzer(tokenized_corpus, shuffle_tokens=shuffle_tokens)
        advanced_analyzer = AdvancedCorpusAnalyzer(tokenized_corpus)

        K_sampling, Beta_sampling = advanced_analyzer.calculate_heaps_law_sampling()
        K, Beta = advanced_analyzer.calculate_heaps_law()

        estimated_vocabulary_size_sampling = K_sampling * (basic_analyzer.total_token_count ** Beta_sampling)
        estimated_vocabulary_size_standard = K * (basic_analyzer.total_token_count ** Beta)

        return estimated_vocabulary_size_sampling, estimated_vocabulary_size_standard

    # Load and tokenize the corpus
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True)
    tokenized_corpus = tokenizer.tokenize(corpus_tokens, lowercase=True)

    # Perform basic analysis
    basic_analyzer = BasicCorpusAnalyzer(tokenized_corpus)
    advanced_analyzer = AdvancedCorpusAnalyzer(tokenized_corpus)
    actual_vocabulary_size = len(basic_analyzer.frequency)
    hapax_legomena_count = sum(1 for _, details in basic_analyzer.token_details.items() if details['frequency'] == 1)

    logger.info(f"\nAnalyzing {corpus_name} Corpus")
    logger.info(f"Total Token Count: {basic_analyzer.total_token_count}")
    logger.info(f"Total Word Types (Distinct Tokens): {actual_vocabulary_size}")
    logger.info(f"Total Hapax Legomena (Unique Tokens): {hapax_legomena_count}")
    logger.info(f"Total Token Count: {basic_analyzer.total_token_count}")
    logger.info(f"Total Word Types (Distinct Tokens): {actual_vocabulary_size}")
    logger.info(f"Total Hapax Legomena (Unique Tokens): {hapax_legomena_count}")
    logger.info(f"Yule's K: {advanced_analyzer.yules_k():.6f}")
    logger.info(f"Herdan's C: {advanced_analyzer.herdans_c():.6f}")
    logger.info(f"Zipf Alpha: {advanced_analyzer.calculate_alpha():.6f}")

    # Perform Heaps' Law analysis for both shuffled and unshuffled tokens
    estimated_vocabulary_size_sampling_unshuffled, estimated_vocabulary_size_standard_unshuffled = heaps_law_analysis(False)
    estimated_vocabulary_size_sampling_shuffled, estimated_vocabulary_size_standard_shuffled = heaps_law_analysis(True)

    # Determine which estimation is closest to the actual vocabulary size
    estimations = {
        'Unshuffled Sampling': estimated_vocabulary_size_sampling_unshuffled,
        'Unshuffled Standard': estimated_vocabulary_size_standard_unshuffled,
        'Shuffled Sampling': estimated_vocabulary_size_sampling_shuffled,
        'Shuffled Standard': estimated_vocabulary_size_standard_shuffled
    }
    closest_estimation = min(estimations, key=lambda x: abs(estimations[x] - actual_vocabulary_size))
    logger.info(f"Most Accurate Estimation: {closest_estimation}")

# Analyze multiple corpora
corpora = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']
for corpus in corpora:
    analyze_corpus(corpus)
