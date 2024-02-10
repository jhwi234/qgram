import numpy as np
import kenlm
import logging
import subprocess
from pathlib import Path
import nltk
from typing import Set
import re
from concurrent.futures import ProcessPoolExecutor


# Precompile the regex to remove non-alphabetic characters
regex = re.compile('[^a-zA-Z]')

class Config:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir else Path(__file__).parent)
        self.model_dir = self.base_dir / "entropy_model"
        self.q_gram = 6  # Modify as needed
        self.model_dir.mkdir(parents=True, exist_ok=True)

# Configure logging at the start of the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command, error_message):
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False
    return True

def build_kenlm_model(corpus_name, q, corpus_path, model_directory):
    arpa_file = model_directory / f"{corpus_name}_{q}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q}gram.klm"
    if run_command(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)], "lmplz failed to generate ARPA model") and \
       run_command(['build_binary', '-s', str(arpa_file), str(binary_file)], "build_binary failed to convert ARPA model to binary format"):
        return str(binary_file)
    return None

def load_corpus(corpus_name: str) -> Set[str]:
    """Load or download a corpus by name using NLTK's direct method access."""
    words = set()
    try:
        # Directly access the corpus if it's a standard one in NLTK
        if hasattr(nltk.corpus, corpus_name):
            corpus = getattr(nltk.corpus, corpus_name)
            # Ensure the corpus is downloaded
            try:
                corpus.ensure_loaded()
            except AttributeError:
                # Some corpora like 'words' don't have ensure_loaded; they're always available
                pass
            # Collect words, handling corpora that are structured differently
            if hasattr(corpus, 'words'):
                words = {word.lower() for word in corpus.words()}
            elif corpus_name == 'stopwords':
                # Example for a non-standard structure: stopwords
                for language in corpus.fileids():
                    words.update(corpus.words(language))
        else:
            # Attempt to download if the corpus isn't recognized as a standard attribute
            nltk.download(corpus_name)
            # Retry loading after download
            if hasattr(nltk.corpus, corpus_name):
                corpus = getattr(nltk.corpus, corpus_name)
                words = {word.lower() for word in corpus.words()}
            else:
                logging.error(f"Downloaded {corpus_name} but could not load it.")
    except Exception as e:
        logging.error(f"Failed to load or download corpus {corpus_name}: {e}")
    
    return words

def generate_formatted_corpus(data_set, formatted_corpus_path):
    """Generate a formatted corpus from a dataset, filtering out short words."""
    # Apply regex substitution once per word and filter in a single pass
    formatted_text = [
        ' '.join(cleaned_word) for word in data_set
        if (cleaned_word := regex.sub('', word)) and len(cleaned_word) >= 4
    ]
    # Write the formatted text to the file
    with formatted_corpus_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(formatted_text))

def calculate_entropy(model, words):
    """Calculate the average entropy for a list of words, reflecting the concept of conditional probability and position-specific entropy."""
    if not words:
        logging.warning("No words provided for entropy calculation.")
        return float('nan')
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    entropies = []  # Store entropy for each position in each word

    # Precompute the join operation for the alphabet to speed up the loop
    alphabet_space_joined = [' '.join(char) for char in alphabet]
    
    for word in words:
        # Convert word to a format compatible with the model scoring function
        spaced_word = ' '.join(word)
        for position in range(len(word)):
            # Generate variants with one character replaced
            variants = [(spaced_word[:2*position] + char + spaced_word[2*position+1:]) for char in alphabet_space_joined]
            # Calculate log probabilities for all variants in one go, if supported by the model API
            log_probs = np.array([model.score(variant, bos=False, eos=False) for variant in variants])
            # Convert log probabilities to probabilities, ensuring numerical stability
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= probs.sum()  # Normalize probabilities
            # Calculate and accumulate entropy for this position
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)
    
    # Return the mean entropy across all positions and words
    return np.mean(entropies) if entropies else float('nan')

def main():
    config = Config()
    corpora = ['brown', 'gutenberg', 'reuters', 'webtext']

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
