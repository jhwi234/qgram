import numpy as np
import kenlm
import logging
import subprocess
from pathlib import Path
import pandas as pd
import regex
import math
from collections import Counter

# Define global configurations for the script
CORPUS_PATH = Path.cwd() / "data/corpora/Linear_B_Lexicon_Final_Cleaned.csv"
MODEL_DIR = Path.cwd() / "entropy_model"
Q_GRAM = 6  # The n-gram level for the KenLM model

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_command(command, error_message):
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        logging.info("Command executed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False
    return True

def build_kenlm_model(corpus_path, model_directory, q_gram):
    corpus_name = corpus_path.stem
    arpa_file = model_directory / f"{corpus_name}_{q_gram}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"
    
    lmplz_command = f"lmplz -o {q_gram} < '{corpus_path}' > '{arpa_file}'"
    build_binary_command = f"build_binary -s '{arpa_file}' '{binary_file}'"
    
    if run_command(lmplz_command, "lmplz failed to generate ARPA model") and run_command(build_binary_command, "build_binary failed to convert ARPA model to binary format"):
        logging.info(f"Model built successfully: {binary_file}")
        return binary_file
    else:
        return None

def load_and_format_corpus(csv_path):
    df = pd.read_csv(csv_path)
    formatted_corpus_path = csv_path.with_suffix('.txt')
    
    with formatted_corpus_path.open('w', encoding='utf-8') as f:
        for word in df['word']:
            formatted_word = ' '.join(regex.findall(r'\X', word))
            f.write(formatted_word + '\n')

    return formatted_corpus_path

def calculate_entropy_kenlm(model, text):
    prepared_text = ' '.join(text)
    log_prob = model.score(prepared_text, bos=True, eos=True)
    log_prob /= math.log10(2)  # Convert to log2
    num_trigrams = len(text) - 2
    return -log_prob / num_trigrams

def calculate_redundancy(H, H_max):
    redundancy = (1 - H / H_max) * 100  # Multiply by 100 to get a percentage
    return redundancy

def process_linearb_corpus(corpus_path, q_gram):
    formatted_corpus_path = load_and_format_corpus(corpus_path)
    model_path = build_kenlm_model(formatted_corpus_path, MODEL_DIR, q_gram)
    
    if model_path:
        model = kenlm.Model(str(model_path))  # Convert model_path to string
        formatted_text = formatted_corpus_path.read_text(encoding='utf-8')
        H0 = math.log2(len(set(formatted_text.replace(' ', ''))))
        
        # Calculate H1 based on character frequencies
        letter_freq = Counter(char for char in formatted_text if char.strip() and char != ' ')
        total_letters = sum(letter_freq.values())
        H1 = -sum((freq / total_letters) * math.log2(freq / total_letters) for freq in letter_freq.values())
        
        # H3 using KenLM
        H3_kenlm = calculate_entropy_kenlm(model, formatted_text)
        
        redundancy = calculate_redundancy(H3_kenlm, H0)
        
        logging.info(f"Linear B Corpus")
        logging.info(f"Zero-order approximation (H0): {H0:.2f}")
        logging.info(f"First-order approximation (H1): {H1:.2f}")
        logging.info(f"Third-order approximation of entropy (H3) with KenLM: {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")

if __name__ == '__main__':
    process_linearb_corpus(CORPUS_PATH, Q_GRAM)