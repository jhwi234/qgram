import numpy as np
import kenlm
import logging
import subprocess
from pathlib import Path
import nltk
import re

# Configure logging at the start of the script
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Precompile the regex to remove non-alphabetic characters
regex = re.compile('[^a-zA-Z]')

class Config:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir else Path(__file__).parent)
        self.model_dir = self.base_dir / "entropy_model"
        self.q_gram = 6
        self.model_dir.mkdir(parents=True, exist_ok=True)

def run_command(command, error_message):
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False

def build_and_load_corpus(corpus_name, config):
    corpus_path = config.model_dir / f"{corpus_name}_formatted.txt"
    if not corpus_path.exists():
        try:
            nltk.download(corpus_name, quiet=True)
            corpus = getattr(nltk.corpus, corpus_name, None)
            if corpus:
                with corpus_path.open('w', encoding='utf-8') as f:
                    for word in corpus.words():
                        cleaned_word = regex.sub('', word.lower())
                        if len(cleaned_word) >= 4:
                            f.write(' '.join(cleaned_word) + '\n')
            else:
                logging.error(f"Corpus {corpus_name} is not available.")
        except Exception as e:
            logging.error(f"Failed to load or download corpus {corpus_name}: {e}")
    return corpus_path

def calculate_entropy(model_path, corpus_path):
    model = kenlm.Model(str(model_path))
    with corpus_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    scores = np.array([model.score(line.strip(), bos=False, eos=False) for line in lines])
    probs = np.exp(scores - np.max(scores))
    entropy = -np.sum(probs * np.log(probs) / len(scores))
    return np.mean(entropy)

def main():
    config = Config()
    corpora = ['brown', 'gutenberg', 'reuters', 'webtext']

    for corpus_name in corpora:
        corpus_path = build_and_load_corpus(corpus_name, config)
        model_path = config.model_dir / f"{corpus_name}_{config.q_gram}gram.klm"
        if not model_path.exists():
            if run_command(['lmplz', '--discount_fallback', '-o', str(config.q_gram), '--text', str(corpus_path), '--arpa', str(corpus_path.with_suffix('.arpa'))], "lmplz failed to generate ARPA model") and \
               run_command(['build_binary', '-s', str(corpus_path.with_suffix('.arpa')), str(model_path)], "build_binary failed to convert ARPA model to binary format"):
                logging.info(f"Model built for {corpus_name}")
        if model_path.exists():
            average_entropy = calculate_entropy(model_path, corpus_path)
            logging.info(f'Average entropy for {corpus_name}: {average_entropy}')

if __name__ == '__main__':
    main()
