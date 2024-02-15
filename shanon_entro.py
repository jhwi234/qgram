import logging
import math
import subprocess
from collections import Counter
from pathlib import Path

import kenlm
import nltk
import regex as reg

# Configuration
Q_GRAMS = 8
logging.basicConfig(level=logging.INFO, format='%(message)s')

def ensure_directory_exists(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def ensure_corpus_available(corpus_name):
    nltk.download(corpus_name, quiet=True)

def prepare_text_for_kenlm(words, corpus_name, corpora_dir="entropy_corpora"):
    ensure_directory_exists(corpora_dir)
    file_path = Path(corpora_dir) / f"{corpus_name}_formatted_corpus.txt"
    cleaned_words = [' '.join(reg.sub(r'[^a-zA-Z]', '', word).lower()) for word in words if len(word) >= 3]
    formatted_text = '\n'.join(cleaned_words)
    
    with file_path.open('w', encoding='utf-8') as f:
        f.write(formatted_text)

    return file_path

def train_kenlm_model(text, n, model_name, model_dir):
    ensure_directory_exists(model_dir)
    model_path = Path(model_dir) / f"{model_name}.klm"
    text_path = model_path.with_suffix('.txt')
    
    with text_path.open('w') as f:
        f.write(text)

    # Redirecting output to DEVNULL to clean up the logging output
    subprocess.run(f"lmplz -o {n} --discount_fallback < {text_path} > {text_path.with_suffix('.arpa')}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(f"build_binary {text_path.with_suffix('.arpa')} {model_path}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    text_path.unlink(missing_ok=True)
    text_path.with_suffix('.arpa').unlink(missing_ok=True)

    return str(model_path)

def calculate_entropy_kenlm(model, lines):
    prepared_text = ' '.join(lines)
    log_prob = model.score(prepared_text, bos=False, eos=False)
    log_prob /= math.log(2)
    num_grams = len(prepared_text.split()) - Q_GRAMS
    return -log_prob / num_grams

def process_corpora(corpus):
    for corpus_name in corpus:
        ensure_corpus_available(corpus_name)
        words = getattr(nltk.corpus, corpus_name).words()
        
        formatted_corpus_path = prepare_text_for_kenlm(words, corpus_name)
        text_for_kenlm = Path(formatted_corpus_path).read_text(encoding='utf-8')
        
        model_path = train_kenlm_model(text_for_kenlm, Q_GRAMS, f"{corpus_name}_char_model", "entropy_model")
        model = kenlm.Model(model_path)
        
        alphabet = sorted(set(text_for_kenlm.replace('\n', '').replace(' ', '')))
        H0 = math.log2(len(alphabet))
        H3_kenlm = calculate_entropy_kenlm(model, text_for_kenlm)
        redundancy = (1 - H3_kenlm / H0) * 100

        #Calculating H1 which is the unigram frequency for each letter
        unigram_freq = Counter(text_for_kenlm.replace('\n', '').replace(' ', ''))
        total_unigrams = sum(unigram_freq.values())
        H1 = -sum((freq / total_unigrams) * math.log2(freq / total_unigrams) for freq in unigram_freq.values())

        # Use logging.info for cleaner output
        logging.info(f"\nCorpus: {corpus_name}")
        logging.info(f"Token Count: {len(words)}")
        logging.info(f"Vocab Count: {len(set(words))}")
        logging.info(f"Alphabet Size: {len(alphabet)}")
        logging.info(f"Zero-order approximation (H0): {H0:.2f}")
        logging.info(f'First-order approximation (H1): {H1:.2f}')
        logging.info(f"Third-order approximation (H3) of {Q_GRAMS}-grams: {H3_kenlm:.2f}")
        logging.info(f"Redundancy: {redundancy:.2f}%")

        Path(formatted_corpus_path).unlink(missing_ok=True)

process_corpora(['brown', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'state_union', 'gutenberg'])
