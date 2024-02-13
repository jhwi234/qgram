import numpy as np
import kenlm
import logging
import subprocess
from pathlib import Path
import pandas as pd
import regex  # Using regex for improved Unicode support

# Define global configurations for the script
CORPUS_PATH = Path.cwd() / "data/corpora/Linear_B_Lexicon_Final_Cleaned.csv"
MODEL_DIR = Path.cwd() / "data/corpora/entropy_model"
Q_GRAM = 6  # Adjust based on requirements

MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Ensure model directory exists
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_command(command, error_message):
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()} (Exit code: {e.returncode})")
        return False

def build_kenlm_model(corpus_path, model_directory, q_gram):
    """Builds and returns path to KenLM model if successful."""
    corpus_name = corpus_path.stem
    arpa_file = model_directory / f"{corpus_name}_{q_gram}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q_gram}gram.klm"
    
    lmplz_command = ['lmplz', '--discount_fallback', '-o', str(q_gram), '--text', str(corpus_path), '--arpa', str(arpa_file)]
    build_binary_command = ['build_binary', '-s', str(arpa_file), str(binary_file)]
    
    if run_command(lmplz_command, "lmplz failed to generate ARPA model") and run_command(build_binary_command, "build_binary failed to convert ARPA model to binary format"):
        return str(binary_file)
    return None

def load_and_format_corpus(csv_path):
    """Formats Linear B corpus, ensuring Unicode characters are preserved and inserts space between characters."""
    df = pd.read_csv(csv_path)
    # Use regex to preserve Linear B characters and insert space between each character
    formatted_words = df['word'].apply(lambda x: ' '.join(regex.sub(r'[^\p{IsLinearB}]+', '', x)))
    formatted_corpus_path = csv_path.with_suffix('.formatted.txt')
    
    # Write all formatted words directly without filtering
    formatted_words.to_csv(formatted_corpus_path, index=False, header=False, encoding='utf-8')
    return formatted_corpus_path

def calculate_entropy(model_path, words):
    """Calculates and returns the average entropy of the corpus based on the KenLM model."""
    model = kenlm.Model(model_path)
    # Ensure each item in 'words' is treated as a string, and filter out any non-string or empty values
    valid_words = [word for word in words if isinstance(word, str) and word.strip()]
    entropies = np.array([model.score(' '.join(word), bos=False, eos=False) for word in valid_words])
    probs = np.exp(entropies - np.max(entropies))
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs) / probs.sum())
    return entropy

def predict_correct_letter_probability(model, corpus_path):
    """
    Estimates the probability of correctly predicting a given letter from the corpus,
    based on the KenLM model's unigram probabilities.
    """
    # Load the corpus to get a set of unique characters
    with open(corpus_path, 'r', encoding='utf-8') as file:
        unique_chars = set(regex.sub(r'[^\p{IsLinearB}]+', '', file.read()))

    # Calculate unigram probabilities from the KenLM model
    unigram_probs = {char: 10**model.score(f' {char} ', bos=False, eos=False) for char in unique_chars}

    # Normalize these probabilities to ensure they sum to 1
    total_prob = sum(unigram_probs.values())
    normalized_probs = {char: prob / total_prob for char, prob in unigram_probs.items()}

    # The probability of correctly predicting a given letter is the sum of squared probabilities,
    # as guessing according to the distribution maximizes the chance of being correct.
    correct_prediction_prob = sum(prob**2 for prob in normalized_probs.values())

    return correct_prediction_prob

def main():
    """Orchestrates corpus formatting, model building, entropy calculation, and letter prediction probability."""
    formatted_corpus_path = load_and_format_corpus(CORPUS_PATH)
    model_path = build_kenlm_model(formatted_corpus_path, MODEL_DIR, Q_GRAM)
    
    if model_path:
        model = kenlm.Model(model_path)
        
        # Inform about the model path
        logging.info(f"KenLM Model Path: {model_path}")
        
        # Load words from formatted corpus and ensure all are treated as strings
        words_df = pd.read_csv(formatted_corpus_path, header=None, encoding='utf-8').dropna()
        words = words_df.iloc[:, 0].apply(str).tolist()
        unique_words = set(words)
        all_characters = set(''.join(unique_words))  # Now guaranteed to only have strings

        num_words = len(words)
        num_unique_words = len(unique_words)
        num_distinct_characters = len(all_characters)
        
        logging.info(f'Token Count: {num_words}')
        logging.info(f'Type Count: {num_unique_words}')
        logging.info(f'Character Count: {num_distinct_characters}')
        
        # Calculate and log average entropy
        average_entropy = calculate_entropy(model_path, words)
        logging.info(f'Average Letter Entropy: {average_entropy:.6f} bits')
        
        # Calculate and log correct letter prediction probability
        correct_letter_prob = predict_correct_letter_probability(model, CORPUS_PATH)
        logging.info(f'Average Letter Probability: {correct_letter_prob:.6%}')

if __name__ == '__main__':
    main()

