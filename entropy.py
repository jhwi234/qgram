import numpy as np
import kenlm
import logging
import subprocess
from pathlib import Path
import nltk
import re

regex = re.compile('[^a-zA-Z]')

class Config:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir else Path(__file__).parent)
        self.model_dir = self.base_dir / "entropy_model"
        self.q_gram = 6  # Modify as needed
        self.model_dir.mkdir(parents=True, exist_ok=True)

def run_command(command, error_message):
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()}")
        return False
    return True

def build_kenlm_model(corpus_name, q, corpus_path, model_directory):
    arpa_file = model_directory / f"{corpus_name}_{q}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q}gram.klm"
    if run_command(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)], "lmplz failed to generate ARPA model") and \
       run_command(['build_binary', '-s', str(arpa_file), str(binary_file)], "build_binary failed to convert ARPA model to binary format"):
        return str(binary_file)
    return None

def load_corpus(corpus_name):
    try:
        nltk.download(corpus_name, quiet=True)
        words = set(w.lower() for w in nltk.corpus.__getattr__(corpus_name).words())
    except AttributeError:
        corpus_path = Config().base_dir / "corpora" / f"{corpus_name}.txt"
        with open(corpus_path, 'r', encoding='utf-8') as file:
            words = set(file.read().lower().split())
    return words

def generate_formatted_corpus(data_set, formatted_corpus_path):
    
    # Filter and format each word in the dataset
    formatted_text = []
    for word in data_set:
        # Remove numbers and punctuation from each word
        cleaned_word = regex.sub('', word)
        # Separate each letter with a space if the word is not empty after cleaning
        if cleaned_word:
            formatted_word = ' '.join(cleaned_word)
            formatted_text.append(formatted_word)
    
    # Write the formatted text to the specified file
    with formatted_corpus_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(formatted_text))

def calculate_entropy(model, words):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    entropies = []
    for word in words:
        for position in range(len(word)):
            log_probs = np.array([model.score(' '.join(word[:position] + char + word[position+1:]), bos=False, eos=False) for char in alphabet])
            probs = np.exp(log_probs - np.max(log_probs))  # Normalize to prevent underflow
            probs /= probs.sum()  # Ensure it sums to 1
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)
    return np.mean(entropies)

def main():
    config = Config()
    corpora = ['brown', 'cmudict', 'CLMET3.txt', 'gutenberg', 'reuters', 'webtext']

    for corpus_name in corpora:
        words = load_corpus(corpus_name)
        formatted_corpus_path = config.model_dir / f"{corpus_name}_formatted.txt"
        generate_formatted_corpus(words, formatted_corpus_path)
        
        model_path = build_kenlm_model(corpus_name, config.q_gram, formatted_corpus_path, config.model_dir)
        if model_path:
            model = kenlm.Model(model_path)
            average_entropy = calculate_entropy(model, [' '.join(word) for word in words])
            print(f'Average entropy for {corpus_name}: {average_entropy}')

if __name__ == '__main__':
    main()
