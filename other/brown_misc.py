import nltk
from nltk.corpus import brown, gutenberg, reuters, inaugural, webtext

def calculate_no_more_than_three_times_percentage(corpus):
    try:
        words = corpus.words()
        freq_dist = nltk.FreqDist(words)
        
        # Words that occur no more than three times
        no_more_than_three_times = sum(count for word, count in freq_dist.items() if count <= 3)
        
        # Calculate percentage
        no_more_than_three_times_percentage = (no_more_than_three_times / len(words)) * 100
        
        return no_more_than_three_times_percentage
    except Exception as e:
        return str(e)

# Mapping of corpus objects to their names
corpora_to_try = {
    'brown': brown,
    'inaugural': inaugural,
    'webtext': webtext,
    'gutenberg': gutenberg,
    'reuters': reuters
}

# Variables to store total percentages for averaging
total_percentage = 0

# Process each corpus, print the percentage, and accumulate for average calculation
for corpus_name, corpus in corpora_to_try.items():
    percentage = calculate_no_more_than_three_times_percentage(corpus)
    print(f"{corpus_name}: {percentage:.2f}% of tokens appear no more than three times")
    total_percentage += percentage

# Calculate and print the average percentage across all corpora
average_percentage = total_percentage / len(corpora_to_try)
print(f"\nAverage across all corpora: {average_percentage:.2f}% of tokens appear no more than three times")
