import math
from collections import Counter
from pathlib import Path
import pandas as pd
import regex
import kenlm
import logging
import subprocess

# Configuration
CORPUS_PATH = Path.cwd() / "data/corpora/Linear_B_Lexicon.csv"
MODEL_DIR = Path.cwd() / "entropy_model"
Q_GRAMS = 4  # KenLM model n-gram level

# Setup
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_command(command, error_message):
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False
    return True

def build_kenlm_model(corpus_path, model_directory, q_gram):
    corpus_name = corpus_path.stem
    arpa_file = model_directory / f"{corpus_name}_{q_gram}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"
    
    if run_command(f"lmplz -o {q_gram} --text {corpus_path} --arpa {arpa_file}", "Failed to generate ARPA model") and \
       run_command(f"build_binary {arpa_file} {binary_file}", "Failed to convert ARPA model to binary format"):
        return binary_file

def load_and_format_corpus(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure processing unique words only to avoid duplicate processing
    unique_words_series = df['word'].drop_duplicates()

    formatted_corpus_path = csv_path.with_suffix('.txt')
    
    with formatted_corpus_path.open('w', encoding='utf-8') as f:
        for word in unique_words_series:
            # Adjust regex to match expected word patterns, considering Linear B script characteristics
            formatted_word = ' '.join(regex.findall(r'[\U00010000-\U0001007F\U00010080-\U000100FF]', word))
            # Ensure that formatted_word is not empty to avoid blank lines
            if formatted_word.strip():
                f.write(formatted_word + '\n')
    
    # Counting unique words after potentially filtering out any that result in blank formatting
    unique_words = len(unique_words_series)

    return formatted_corpus_path, unique_words

def calculate_entropy_kenlm(model, lines):
    prepared_text = ' '.join(lines)
    log_prob = model.score(prepared_text, bos=False, eos=False)
    log_prob /= math.log(2)
    num_grams = len(prepared_text.split()) - Q_GRAMS
    return -log_prob / num_grams

def calculate_redundancy(H, H_max):
    return (1 - H / H_max) * 100

def process_linearb_corpus(corpus_path, q_gram):
    formatted_corpus_path, unique_words = load_and_format_corpus(corpus_path)
    model_path = build_kenlm_model(formatted_corpus_path, MODEL_DIR, q_gram)
    
    if model_path:
        model = kenlm.Model(str(model_path))
        lines = Path(formatted_corpus_path).read_text(encoding='utf-8').split('\n')
        H0 = math.log2(len(set(''.join(lines).replace(' ', ''))))
        letter_freq = Counter(''.join(lines).replace(' ', ''))
        total_letters = sum(letter_freq.values())
        H1 = -sum((freq / total_letters) * math.log2(freq / total_letters) for freq in letter_freq.values())
        H3_kenlm = calculate_entropy_kenlm(model, lines)
        redundancy = calculate_redundancy(H3_kenlm, H0)
        
        logging.info(f"Linear B Corpus")
        logging.info(f"Vocab Count: {unique_words}")
        logging.info(f'Alphabet Size: {len(letter_freq):,}')
        logging.info(f"Zero-order approximation (H0): {H0:.2f}")
        logging.info(f"First-order approximation (H1): {H1:.2f}")
        logging.info(f"Third-order approximation (H3): {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")

if __name__ == '__main__':
    process_linearb_corpus(CORPUS_PATH, Q_GRAMS)