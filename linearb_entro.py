import numpy as np
import kenlm
import logging
import subprocess
from pathlib import Path
import pandas as pd
import regex

# Define global configurations for the script
CORPUS_PATH = Path.cwd() / "data/corpora/Linear_B_Lexicon_Final_Cleaned.csv"
MODEL_DIR = Path.cwd() / "data/corpora/entropy_model"
Q_GRAM = 6  # Number of grams for KenLM model

# Ensure the model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command, error_message):
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False

def build_kenlm_model(corpus_path, model_directory, q_gram):
    corpus_name = corpus_path.stem  # Use the stem of the corpus file path as the model name
    arpa_file = model_directory / f"{corpus_name}_{q_gram}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"
    if run_command(['lmplz', '--discount_fallback', '-o', str(q_gram), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                   "lmplz failed to generate ARPA model") and \
       run_command(['build_binary', '-s', str(arpa_file), str(binary_file)],
                   "build_binary failed to convert ARPA model to binary format"):
        return str(binary_file)
    return None

def load_and_format_corpus(csv_path):
    """Load and format the Linear B corpus from a CSV file for KenLM."""
    df = pd.read_csv(csv_path)
    # Clean and lower-case words, removing non-alphabetic characters
    formatted_words = df['word'].apply(lambda x: regex.sub(r'\P{L}+', '', x).lower())
    formatted_corpus_path = csv_path.with_suffix('.formatted.txt')
    formatted_words.to_csv(formatted_corpus_path, index=False, header=False)
    return formatted_corpus_path

def calculate_entropy(model_path, words):
    """Calculate the entropy of the corpus based on the KenLM model."""
    model = kenlm.Model(model_path)
    entropies = [model.score(' '.join(word), bos=False, eos=False) for word in words]
    log_probs = np.array(entropies)
    probs = np.exp(log_probs - np.max(log_probs))
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def main():
    formatted_corpus_path = load_and_format_corpus(CORPUS_PATH)
    model_path = build_kenlm_model(formatted_corpus_path, MODEL_DIR, Q_GRAM)
    if model_path:
        logging.info(f"KenLM model built at {model_path}")
        # Load formatted words for entropy calculation
        words = pd.read_csv(formatted_corpus_path, header=None).iloc[:, 0].tolist()
        average_entropy = calculate_entropy(model_path, words)
        logging.info(f'Average entropy: {average_entropy}')

if __name__ == '__main__':
    main()
