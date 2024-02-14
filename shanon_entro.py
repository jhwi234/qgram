import nltk
import math
from collections import Counter
import subprocess
from pathlib import Path
import kenlm
import regex as reg

def ensure_directory_exists(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def ensure_corpus_available(corpus_name):
    nltk.download(corpus_name, quiet=True)

def prepare_text_for_kenlm(words, corpus_name, corpora_dir="entropy_corpora"):
    ensure_directory_exists(corpora_dir)  # Ensure the directory exists
    file_path = Path(corpora_dir) / f"{corpus_name}_formatted_corpus.txt"  # Name files based on corpus name
    
    cleaned_words = [reg.sub(r'[^a-zA-Z]', '', word) for word in words if len(reg.sub(r'[^a-zA-Z]', '', word)) >= 3]
    formatted_text = '\n'.join([' '.join(word) for word in cleaned_words])
    
    with file_path.open('w', encoding='utf-8') as f:
        f.write(formatted_text.lower())

    return file_path

def read_prepared_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def train_kenlm_model(text, n=6, model_name="char_trigram_model", model_dir="entropy_model"):
    ensure_directory_exists(model_dir)
    trigrams_path = Path(model_dir) / f"{model_name}.txt"
    arpa_path = Path(model_dir) / f"{model_name}.arpa"
    klm_path = Path(model_dir) / f"{model_name}.klm"

    with trigrams_path.open('w') as f:
        f.write(text)

    subprocess.run(f"lmplz -o {n} --discount_fallback < {trigrams_path} > {arpa_path}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(f"build_binary {arpa_path} {klm_path}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    trigrams_path.unlink()
    arpa_path.unlink()

    return str(klm_path)

def extract_alphabet(text):
    distinct_chars = sorted(set(char for char in text if char.isalpha() or char == ' '))
    return distinct_chars

def calculate_entropy_kenlm(model, text):
    prepared_text = ' '.join(text)
    log_prob = model.score(prepared_text, bos=True, eos=True)
    log_prob /= math.log10(2)  # Convert to log2
    num_trigrams = len(text) - 2
    return -log_prob / num_trigrams

def calculate_redundancy(H, H_max):
    """
    Calculate the redundancy of a language based on the actual entropy (H) and
    the maximum possible entropy (H_max).
    """
    redundancy = (1 - H / H_max) * 100  # Multiply by 100 to get a percentage
    return redundancy

def cleanup_files(*files):
    for file_path in files:
        try:
            Path(file_path).unlink()
        except FileNotFoundError:
            pass

def process_corpora(corpus):
    for corpus_name in corpus:
        ensure_corpus_available(corpus_name)
        corpus = getattr(nltk.corpus, corpus_name)
        
        formatted_corpus_path = prepare_text_for_kenlm(corpus.words(), corpus_name)
        text_for_kenlm = read_prepared_text(formatted_corpus_path)
        
        model_name = f"{corpus_name}_char_trigram_model"
        klm_file_path = train_kenlm_model(text_for_kenlm, 8, model_name)
        model = kenlm.Model(klm_file_path)
        
        letter_freq = Counter(char.lower() for word in corpus.words() for char in word if char.isalpha())
        alphabet = extract_alphabet(text_for_kenlm)
        total_letters = sum(letter_freq[letter] for letter in alphabet)
        
        H0 = math.log2(len(alphabet))
        prob_actual = {letter: freq / total_letters for letter, freq in letter_freq.items() if letter in alphabet}
        H1 = -sum(p * math.log2(p) for p in prob_actual.values())
        
        H3_kenlm = calculate_entropy_kenlm(model, text_for_kenlm)

        redundancy = calculate_redundancy(H3_kenlm, H0)
        
        print(f'\nCorpus: {corpus_name}')
        # print(f'Alphabet: {alphabet}')
        print(f'Zero-order approximation (H0): {H0:.2f}')
        print(f'First-order approximation (H1): {H1:.2f}')
        print(f'Third-order approximation of entropy (H3) with KenLM for character 6-grams: {H3_kenlm:.2f}')
        print(f'Redundancy: {redundancy}')

        cleanup_files(formatted_corpus_path)

# Process multiple corpora including 'brown', 'gutenberg', and 'reuters'
process_corpora(['brown', 'gutenberg', 'reuters', 'webtext'])

