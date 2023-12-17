import nltk
from nltk.corpus import brown
import random
from collections import Counter

def simulate_letter_blanking(words, sample_size=400):
    """
    Simulates the blanking out of a random letter from a given sample of words
    and returns words that contain the blanked-out letter.
    """
    # Flatten the sample into a single string of letters
    letters = ''.join(filter(str.isalpha, ''.join(words)))

    # Choose a random letter position to blank out
    blank_position = random.randint(0, len(letters) - 1)

    # Identify the blanked-out letter
    blanked_letter = letters[blank_position]

    # Find words that contain the blanked-out letter
    blanked_words = [word for word in words if blanked_letter in word]

    return blanked_words

# Step 1: Download the Brown corpus
nltk.download('brown')
all_words = brown.words()

# Initialize a Counter to hold the frequencies of words with a blanked-out letter
word_freq = Counter()

# Step 2: Simulate the process for 100 pages
for i in range(100):
    # Take a sample of 400 words for each page
    page_sample = all_words[i * 400: (i + 1) * 400]

    # Simulate the letter blanking for this page
    blanked_words = simulate_letter_blanking(page_sample)

    # Update the overall word frequency counter
    word_freq.update(blanked_words)

# Step 3: Identify the top ten word types most likely to have a letter blanked out
top_ten_words = word_freq.most_common(10)

print("Top ten words likely to have a letter blanked out in 100 pages:")
for word, freq in top_ten_words:
    print(f"{word}: {freq}")