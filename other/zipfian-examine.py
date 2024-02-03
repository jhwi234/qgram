import nltk
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Function to get cleaned words from a corpus
def get_cleaned_words(corpus_name):
    nltk.download(corpus_name, quiet=True)
    if isinstance(getattr(nltk.corpus, corpus_name), nltk.corpus.reader.PlaintextCorpusReader):
        corpus_words = getattr(nltk.corpus, corpus_name).words()
    else:
        corpus_files = getattr(nltk.corpus, corpus_name).fileids()
        corpus_words = [word for file_id in corpus_files for word in getattr(nltk.corpus, corpus_name).words(file_id)]
    return [word.lower() for word in corpus_words if word.isalpha()]

# Function to get word types before the median token
def get_word_types_before_median(corpus_name):
    cleaned_words = get_cleaned_words(corpus_name)
    word_frequencies = Counter(cleaned_words)
    total_word_count = sum(word_frequencies.values())

    sorted_word_freq = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    cumulative_count = 0
    word_types_before_median = set()

    for word, count in sorted_word_freq:
        cumulative_count += count
        word_types_before_median.add(word)
        if cumulative_count >= total_word_count / 2:
            break

    return word_types_before_median

# Function to calculate Zipfian probabilities
def calculate_zipfian_probabilities(corpus_name):
    cleaned_words = get_cleaned_words(corpus_name)
    word_frequencies = Counter(cleaned_words)
    total_word_count = sum(word_frequencies.values())

    # Sort words by frequency
    sorted_words = sorted(word_frequencies, key=word_frequencies.get, reverse=True)
    cumulative_count = 0
    top_half_word_types = 0

    for word in sorted_words:
        cumulative_count += word_frequencies[word]
        top_half_word_types += 1
        if cumulative_count >= total_word_count / 2:
            break

    probability_top_half = cumulative_count / total_word_count
    probability_bottom_half = 1 - probability_top_half
    bottom_half_word_types = len(word_frequencies) - top_half_word_types
    return probability_top_half, probability_bottom_half, top_half_word_types, bottom_half_word_types, total_word_count, len(word_frequencies)

# Function to format output
def format_output(corpus, median_prob, complement_median_prob, median_types, complement_median_types, total_tokens, unique_types):
    output = (
        f"Corpus: {corpus}\n"
        f"--------------------------------------------\n"
        f"Probability Before Median Token: {median_prob:.4f}\n"
        f"Probability After Median Token:  {complement_median_prob:.4f}\n"
        f"Word Types Before Median Token:  {median_types}\n"
        f"Word Types After Median Token:   {complement_median_types}\n"
        f"Word Type Count:                 {unique_types}\n"
        f"Word Token Count:                {total_tokens}\n"
    )
    return output

# Function to generate and save a table as a PNG image
def save_table_as_png(data, filename='corpus_analysis.png'):
    plt.figure(figsize=(12, len(data) * 0.625))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = ax.table(cellText=data.values,
                     colLabels=data.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# List of corpora to analyze
corpora_to_analyze = ['brown', 'gutenberg', 'reuters', 'webtext', 'inaugural', 'nps_chat', 'state_union']

# Accumulate results in a DataFrame
results = []
for corpus in corpora_to_analyze:
    top_half_prob, bottom_half_prob, top_half_types, bottom_half_types, unique_types, total_tokens = calculate_zipfian_probabilities(corpus)
    results.append([corpus, top_half_types, bottom_half_types, unique_types, total_tokens])
    print(format_output(corpus, top_half_prob, bottom_half_prob, top_half_types, bottom_half_types, unique_types, total_tokens))

# Create DataFrame
df = pd.DataFrame(results, columns=[
    'Corpus',
    'Word Types Before Median Token',
    'Word Types After Median Token',
    'Word Type Count',
    'Word Token Count'
])

# Sort DataFrame and calculate averages
df = df.sort_values(by='Word Token Count', ascending=False)
average_data = pd.DataFrame([['Average',
                              df['Word Types Before Median Token'].mean(),
                              df['Word Types After Median Token'].mean(),
                              df['Word Type Count'].mean(),
                              df['Word Token Count'].mean()]],
                            columns=df.columns)
df = pd.concat([df, average_data], ignore_index=True)

# Save DataFrame as a PNG table
save_table_as_png(df)

# Get word types before median for each corpus and calculate their intersection
word_types_before_median_sets = [get_word_types_before_median(corpus) for corpus in corpora_to_analyze]
intersection_word_types = set.intersection(*word_types_before_median_sets)

# save intersection word types to a file
with open('intersection_word_types.txt', 'w') as f:
    f.write('\n'.join(intersection_word_types))

# Function to calculate and print the percentage of tokens attributed to the intersection in each corpus
def calculate_and_print_intersection_percentage(corpus_name, intersection_words):
    cleaned_words = get_cleaned_words(corpus_name)
    word_frequencies = Counter(cleaned_words)
    intersection_count = sum(word_frequencies[word] for word in intersection_words)
    total_count = sum(word_frequencies.values())
    percentage = (intersection_count / total_count) * 100
    print(f"{corpus_name}: {percentage:.2f}% of tokens are from the intersection word types")

# Calculate and print percentages for each corpus
for corpus in corpora_to_analyze:
    calculate_and_print_intersection_percentage(corpus, intersection_word_types)
