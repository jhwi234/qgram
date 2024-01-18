import nltk
from nltk.corpus import brown, gutenberg, reuters, webtext, inaugural
import numpy as np

def analyze_corpus(corpus_name: str):
    try:
        corpora = {
            "brown": brown,
            "gutenberg": gutenberg,
            "reuters": reuters,
            "webtext": webtext,
            "inaugural": inaugural
        }
        corpus = corpora[corpus_name]

        sentence_lengths = [len(sentence) for sentence in corpus.sents()]
        average_sentence_length = np.mean(sentence_lengths)
        std_dev_sentence_length = np.std(sentence_lengths)

        word_tokens = corpus.words()
        word_token_lengths = [len(word) for word in word_tokens]
        average_word_token_length = np.mean(word_token_lengths)
        std_dev_word_token_length = np.std(word_token_lengths)

        word_types = set(word_tokens)
        word_type_lengths = [len(word) for word in word_types]
        average_word_type_length = np.mean(word_type_lengths)
        std_dev_word_type_length = np.std(word_type_lengths)

        # Return formatted string
        return f"{corpus_name}\t{average_sentence_length:.2f}\t{average_word_token_length:.2f}\t{average_word_type_length:.2f}\t{std_dev_sentence_length:.2f}\t{std_dev_word_token_length:.2f}\t{std_dev_word_type_length:.2f}"

    except Exception as e:
        return f"Failed to analyze {corpus_name}: {e}"

# Download corpora
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('reuters')
nltk.download('webtext')
nltk.download('inaugural')

# Analyze multiple corpora and collect results
corpora_names = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural']
results = ["Corpus\tAvg. Sentence Length (words)\tAvg. Token Length (letters)\tAvg. Word Type Length (letters)\tStd. Dev. Sentence Length\tStd. Dev. Word Token Length\tStd. Dev. Word Type Length"]
for corpus_name in corpora_names:
    results.append(analyze_corpus(corpus_name))

# Print final output
print("\n".join(results))