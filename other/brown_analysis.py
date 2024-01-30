import nltk
from nltk.corpus import brown, gutenberg, reuters, webtext, inaugural
import numpy as np

def download_if_missing(corpus_name):
    try:
        nltk.data.find('corpora/' + corpus_name)
    except LookupError:
        try:
            nltk.download(corpus_name, quiet=True)
        except Exception as e:
            print(f"Error downloading {corpus_name}: {e}")
            return False
    return True

def analyze_corpus(corpus_name: str):
    if not download_if_missing(corpus_name):
        return f"{corpus_name}\tDownload Failed"

    corpora = {
        "brown": brown,
        "gutenberg": gutenberg,
        "reuters": reuters,
        "webtext": webtext,
        "inaugural": inaugural
    }
    corpus = corpora[corpus_name]

    sentence_lengths = np.array([len(sentence) for sentence in corpus.sents()])
    word_tokens = corpus.words()
    word_token_lengths = np.array([len(word) for word in word_tokens])
    word_types = set(word_tokens)
    word_type_lengths = np.array([len(word) for word in word_types])

    # Calculating averages and standard deviations
    avg_sent_len = np.mean(sentence_lengths)
    std_sent_len = np.std(sentence_lengths)
    avg_token_len = np.mean(word_token_lengths)
    std_token_len = np.std(word_token_lengths)
    avg_type_len = np.mean(word_type_lengths)
    std_type_len = np.std(word_type_lengths)

    return f"{corpus_name}\t{avg_sent_len:.2f}\t{std_sent_len:.2f}\t{avg_token_len:.2f}\t{std_token_len:.2f}\t{avg_type_len:.2f}\t{std_type_len:.2f}"

# Headers for the output table
headers = ["Corpus", "Avg. Sentence Length", "Std. Dev. Sentence Length", "Avg. Word Token Length", "Std. Dev. Word Token Length", "Avg. Word Type Length", "Std. Dev. Word Type Length"]
header_line = "\t".join(headers)

# Analyze the corpora
corpora_names = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']
results = [header_line]
for corpus_name in corpora_names:
    results.append(analyze_corpus(corpus_name))

# Print the formatted results
print("\n".join(results))
