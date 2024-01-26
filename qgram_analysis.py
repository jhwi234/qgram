from collections import Counter
import os
from pathlib import Path

def generate_qgrams(word, max_q=6):
    """Generate q-grams for a given word up to a specified maximum size."""
    return [word[i:i+size] for size in range(1, max_q + 1) for i in range(len(word) - size + 1)]

def read_words_from_file(file_path):
    """Read words from a file and return as a set."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())

def write_qgram_frequencies_to_file(qgram_freq, file_path):
    """Write sorted q-gram frequencies to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for qgram, freq in sorted(qgram_freq.items(), key=lambda item: item[1], reverse=True):
            file.write(f"{qgram}: {freq}\n")

def process_corpus(corpus_name, base_dir):
    """Process each corpus to create q-gram frequency files."""
    sets_dir = Path(base_dir) / 'data' / 'outputs' / 'sets'
    qgrams_dir = Path(base_dir) / 'data' / 'outputs' / 'qgrams'
    qgrams_dir.mkdir(exist_ok=True)

    all_words_file = sets_dir / f'{corpus_name}_all_words.txt'
    train_set_file = sets_dir / f'{corpus_name}_train_set.txt'

    try:
        all_words = read_words_from_file(all_words_file)
        train_set = read_words_from_file(train_set_file)

        test_set = all_words - train_set

        # Generating q-grams for training and test sets
        train_qgrams = Counter(qgram for word in train_set for qgram in generate_qgrams(word))
        test_qgrams = Counter(qgram for word in test_set for qgram in generate_qgrams(word))

        # Writing q-gram frequencies to files
        for dataset_type, qgrams in [('train', train_qgrams), ('test', test_qgrams)]:
            qgrams_file = qgrams_dir / f'{corpus_name}_qgram_{dataset_type}.txt'
            write_qgram_frequencies_to_file(qgrams, qgrams_file)

        print(f'Processed {corpus_name} corpus successfully.')

    except FileNotFoundError as e:
        print(f'Error processing {corpus_name} corpus: {e}')

def main():
    base_dir = os.getcwd()
    corpora = ['brown', 'CLMET3', 'cmudict', 'gutenberg', 'inaugural', 'mega_corpus', 'reuters']
    for corpus in corpora:
        process_corpus(corpus, base_dir)

if __name__ == '__main__':
    main