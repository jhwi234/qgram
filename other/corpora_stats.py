import argparse
import logging
import multiprocessing

import nltk
import numpy as np
from nltk.corpus import brown, gutenberg, inaugural, reuters, webtext
import tabulate


def download_corpus(corpus_name):
    """Downloads the specified NLTK corpus if not already present."""
    try:
        nltk.data.find(f'corpora/{corpus_name}')
    except LookupError:
        try:
            nltk.download(corpus_name, quiet=True)
        except Exception as e:
            logging.error(f"Error downloading {corpus_name}: {e}")
            return False
    return True

def compute_statistics(corpus):
    """Computes various statistics for the given corpus."""
    sentence_lengths = np.array([len(sentence) for sentence in corpus.sents()])
    word_tokens = corpus.words()
    word_token_lengths = np.array([len(word) for word in word_tokens])
    word_types = set(word_tokens)

    return {
        'total_sentences': len(sentence_lengths),
        'total_word_tokens': len(word_tokens),
        'vocabulary_size': len(word_types),
        'type_token_ratio': len(word_types) / len(word_tokens),
        'avg_sent_length': np.mean(sentence_lengths),
        'std_sent_length': np.std(sentence_lengths),
        'avg_token_length': np.mean(word_token_lengths),
        'std_token_length': np.std(word_token_lengths)
    }

def analyze_corpus(corpus_name):
    """Analyzes the specified corpus and returns a summary string."""
    if not download_corpus(corpus_name):
        return f"{corpus_name}\tDownload Failed"

    corpora = {
        "brown": brown,
        "gutenberg": gutenberg,
        "reuters": reuters,
        "webtext": webtext,
        "inaugural": inaugural
    }

    if corpus_name not in corpora:
        return f"{corpus_name}\tUnsupported Corpus"

    stats = compute_statistics(corpora[corpus_name])
    return f"{corpus_name}\t{stats['total_sentences']}\t{stats['total_word_tokens']}\t{stats['vocabulary_size']}\t{stats['type_token_ratio']:.2f}\t{stats['avg_sent_length']:.2f}\t{stats['std_sent_length']:.2f}\t{stats['avg_token_length']:.2f}\t{stats['std_token_length']:.2f}"

def main():
    """Main function to analyze specified corpora and print results."""
    parser = argparse.ArgumentParser(description='Corpus Analysis Script')
    parser.add_argument('-c', '--corpora', nargs='+', default=['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural'], help='List of corpora to analyze')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    headers = ["Corpus", "Total Sentences", "Total Word Tokens", "Vocabulary Size", "Type-Token Ratio", "Avg Sent Len", "SD Sent Len", "Avg Token Len", "SD Token Len"]

    with multiprocessing.Pool() as pool:
        results = pool.map(analyze_corpus, args.corpora)

    # Formatting results using tabulate
    formatted_results = [result.split('\t') for result in results]
    print(tabulate(formatted_results, headers=headers, tablefmt='grid'))

if __name__ == "__main__":
    main()