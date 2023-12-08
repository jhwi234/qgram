import nltk
from string import ascii_lowercase

# Ensure the Brown corpus is downloaded
nltk.download('brown')
brown_words = nltk.corpus.brown.words()

# Convert words to lowercase and create a set of unique words
unique_words = set(word.lower() for word in brown_words)

# Initialize dictionaries to store counts
token_frequency = {letter: 0 for letter in ascii_lowercase}
type_frequency = {letter: 0 for letter in ascii_lowercase}

# Count token frequencies
for word in brown_words:
    for letter in word.lower():
        if letter in ascii_lowercase:
            token_frequency[letter] += 1

# Count type frequencies
for word in unique_words:
    for letter in set(word):
        if letter in ascii_lowercase:
            type_frequency[letter] += 1

# Calculate percentages
total_tokens = len(brown_words)
total_types = len(unique_words)

token_percentage = {letter: (count / total_tokens) * 100 for letter, count in token_frequency.items()}
type_percentage = {letter: (count / total_types) * 100 for letter, count in type_frequency.items()}

# Sort and print the results
print("Token Percentages:")
for letter, percentage in sorted(token_percentage.items(), key=lambda item: item[1], reverse=True):
    print(f"{letter}: {percentage:.2f}%")

print("\nType Percentages:")
for letter, percentage in sorted(type_percentage.items(), key=lambda item: item[1], reverse=True):
    print(f"{letter}: {percentage:.2f}%")

import nltk
from string import ascii_lowercase

# Ensure the Brown corpus is downloaded
nltk.download('brown')
brown_words = nltk.corpus.brown.words()

# Convert words to lowercase and create a set of unique words
unique_words = set(word.lower() for word in brown_words)

# Initialize dictionaries to store counts
token_letter_frequency = {letter: 0 for letter in ascii_lowercase}
type_letter_frequency = {letter: 0 for letter in ascii_lowercase}

# Count letter frequencies by tokens
for word in brown_words:
    for letter in word.lower():
        if letter in ascii_lowercase:
            token_letter_frequency[letter] += 1

# Count letter frequencies by types
for word in unique_words:
    for letter in set(word):
        if letter in ascii_lowercase:
            type_letter_frequency[letter] += 1

# Calculate total number of letters
total_letters_by_tokens = sum(token_letter_frequency.values())
total_letters_by_types = sum(type_letter_frequency.values())

# Calculate percentages
token_letter_percentage = {letter: (count / total_letters_by_tokens) * 100 for letter, count in token_letter_frequency.items()}
type_letter_percentage = {letter: (count / total_letters_by_types) * 100 for letter, count in type_letter_frequency.items()}

# Print the results
print("Token Letter Percentages:")
for letter, percentage in sorted(token_letter_percentage.items(), key=lambda item: item[1], reverse=True):
    print(f"{letter}: {percentage:.2f}%")

print("\nType Letter Percentages:")
for letter, percentage in sorted(type_letter_percentage.items(), key=lambda item: item[1], reverse=True):
    print(f"{letter}: {percentage:.2f}%")
