import logging
import random
import re
from pathlib import Path
import enum
import heapq

import kenlm
import numpy as np
import nltk
import subprocess
from retry import retry

# Constants
VOWELS = 'aeiou'
CONSONANTS = ''.join(set('abcdefghijklmnopqrstuvwxyz') - set(VOWELS))

# Enumeration for letter types
class LetterType(enum.Enum):
    VOWEL = 1
    CONSONANT = 2

logging.basicConfig(level=logging.INFO)
clean_pattern = re.compile(r'\b[a-zA-Z]{4,}\b')

@retry(subprocess.CalledProcessError, tries=3, delay=2)
def model_task(corpus_name, q, corpus_path, model_directory):
    arpa_file = model_directory / f'{corpus_name}_{q}gram.arpa'
    binary_file = model_directory / f'{corpus_name}_{q}gram.klm'

    try:
        subprocess.run(
            ['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ['build_binary', '-s', str(arpa_file), str(binary_file)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return q, str(binary_file)
    except subprocess.CalledProcessError as e:
        logging.error(f"Attempt to generate/load the model for {q}-gram failed: {e}")
        raise

class LanguageModel:
    def __init__(self, q_range=(6, 6), split_config=0.9):
        self.q_range = range(q_range[0], q_range[1] + 1)
        self.model = {}
        self.corpus = set()
        self.test_set = set()
        self.training_set = set()
        self.all_words = set()
        self.split_config = split_config

    def clean_text(self, text: str) -> set[str]:
        return set(clean_pattern.findall(text))

    def load_corpus(self):
        nltk.download('cmudict')
        cmu_words = nltk.corpus.cmudict.words()
        cmu_words = [word.lower() for word in cmu_words]
        self.corpus = self.clean_text(' '.join(cmu_words))

    # Modify the prepare_datasets function
    def prepare_datasets(self, min_word_length=None, max_word_length=None, vowel_replacement_ratio=0.5, consonant_replacement_ratio=0.5):
        # Filter corpus based on word length if specified
        if min_word_length or max_word_length:
            self.corpus = {word for word in self.corpus if (min_word_length or 0) <= len(word) <= (max_word_length or float('inf'))}

        total_size = len(self.corpus)
        training_size = int(total_size * self.split_config)
        shuffled_corpus = list(self.corpus)
        random.shuffle(shuffled_corpus)

        self.training_set = set(shuffled_corpus[:training_size])
        unprocessed_test_set = set(shuffled_corpus[training_size:])
        self.save_set_to_file(self.training_set, "cmu_training_set.txt")

        # Update the test set preparation to include the original word
        self.test_set = {self.replace_random_letter(word, include_original=True, vowel_replacement_ratio=vowel_replacement_ratio, consonant_replacement_ratio=consonant_replacement_ratio) for word in unprocessed_test_set}
        self.save_set_to_file(self.test_set, "formatted_test_set.txt")
        # Combine training and test sets into all_words
        self.all_words = self.training_set.union(self.test_set)

    def prepare_and_save_test_set(self, data_set, file_name):
        formatted_test_set = []
        for word in data_set:
            modified_word, missing_letter = self.replace_random_letter(word)
            # Append a tuple containing the modified word, the missing letter, and the original word
            formatted_test_set.append((modified_word, missing_letter, word))

        self.save_set_to_file(formatted_test_set, file_name)

    def replace_random_letter(self, word, include_original=False, vowel_replacement_ratio=0.5, consonant_replacement_ratio=0.5):
        # Create a dictionary to categorize indices of vowels and consonants in the word.
        categorized_indices = {LetterType.VOWEL: [], LetterType.CONSONANT: []}

        # Loop through each letter in the word.
        for i, letter in enumerate(word):
            # Determine if the letter is a vowel or a consonant.
            letter_type = LetterType.VOWEL if letter in VOWELS else LetterType.CONSONANT
            # Add the index of the letter to the corresponding category (vowel or consonant).
            categorized_indices[letter_type].append(i)

        # Decide whether to replace a vowel or a consonant based on their presence and specified ratios.
        replace_vowel = len(categorized_indices[LetterType.VOWEL]) > 0 and random.random() < vowel_replacement_ratio
        replace_consonant = len(categorized_indices[LetterType.CONSONANT]) > 0 and random.random() < consonant_replacement_ratio

        # If both vowel and consonant are candidates for replacement, randomly choose one.
        if replace_vowel and replace_consonant:
            replace_vowel = random.choice([True, False])
            replace_consonant = not replace_vowel

        # Choose a random index from the selected category (vowel or consonant) for replacement.
        if replace_vowel:
            letter_index = random.choice(categorized_indices[LetterType.VOWEL])
        elif replace_consonant:
            letter_index = random.choice(categorized_indices[LetterType.CONSONANT])
        else:
            # If no vowels or consonants are eligible, pick any random letter in the word.
            letter_index = random.randint(0, len(word) - 1)

        # Store the letter to be replaced and create the modified word with a placeholder '_' at the replaced letter's position.
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        # Return a tuple with the modified word and missing letter. If 'include_original' is True, include the original word as well.
        if include_original:
            return (modified_word, missing_letter, word)
        else:
            return modified_word, missing_letter

    def generate_and_load_models(self):
        training_corpus_path = self.generate_formatted_corpus(self.training_set, 'cmu_formatted_training_corpus.txt')
        self.generate_models_from_corpus(training_corpus_path)

    def generate_formatted_corpus(self, data_set, path):
        formatted_text = [" ".join(word) for word in data_set]
        formatted_corpus = '\n'.join(formatted_text)
        corpus_path = Path(path)
        with corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_corpus)
        return path

    def generate_models_from_corpus(self, corpus_path):
        model_directory = Path('cmu_models')
        model_directory.mkdir(parents=True, exist_ok=True)

        for q in self.q_range:
            _, binary_file = model_task('cmu', q, corpus_path, model_directory)
            if binary_file:
                self.model[q] = kenlm.Model(binary_file)
                logging.info(f"Model for {q}-gram loaded.")

    # weight with higher entropy
    def _predict_missing_letter(self, oov_word):
        # Identify the position of the missing letter (denoted by '_') in the out-of-vocabulary word.
        missing_letter_index = oov_word.index('_')

        # Initialize dictionaries to store the log probabilities of each alphabet letter being the missing one
        # and the entropy weights for each context size (q) in the q-gram models.
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyz'}
        entropy_weights = []

        # Add start (<s>) and end (</s>) boundary markers to the word, to simulate sentence boundaries as in the training data.
        oov_word_with_boundaries = f"<s> {oov_word} </s>"

        # Loop through each q-gram model. The q-gram size determines how many letters (context) are considered around the missing letter.
        for q in self.q_range:
            if q not in self.model:
                continue  # Skip if the model for the current q-gram size is not available.

            model = self.model[q]

            # Determine the number of letters (context size) to consider to the left and right of the missing letter.
            # This context size depends on the position of the missing letter and the size of the q-gram.
            left_size = min(missing_letter_index + 1, q - 1)  # Adjust for <s> at the start.
            right_size = min(len(oov_word) - missing_letter_index, q - 1)  # <s> at the end doesn't affect right size.

            # Extract left and right contexts from the word, including the boundary markers.
            left_context = oov_word_with_boundaries[max(4, missing_letter_index - left_size + 4):missing_letter_index + 4]  # 4 accounts for the length of "<s> ".
            right_context = oov_word_with_boundaries[missing_letter_index + 5:missing_letter_index + 5 + right_size]  # 5 accounts for "<s> " and the '_' character.

            # Remove any extra spaces from the context strings.
            left_context = left_context.strip()
            right_context = right_context.strip()

            # Combine the contexts into single strings, as they would appear in natural text.
            left_context_joined = ' '.join(left_context)
            right_context_joined = ' '.join(right_context)

            # Calculate the entropy of the current context. Entropy here measures the uncertainty of the context.
            # It's used to give more weight to more informative contexts in the final prediction.
            entropy = -sum(model.score(left_context_joined + ' ' + c + ' ' + right_context_joined)
                        for c in 'abcdefghijklmnopqrstuvwxyz')
            entropy_weights.append(entropy)

            # For each possible letter, calculate the log probability of it being the correct one to fill the missing spot.
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                # Create a sequence by inserting each potential letter into the placeholder and score it using the model.
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                log_prob_full = model.score(full_sequence)
                # Store the log probability for this letter and context combination.
                log_probabilities[letter].append(log_prob_full)

            # Normalize the entropy weights so they sum to 1, making them proper probabilities.
            entropy_weights = np.exp(entropy_weights - np.max(entropy_weights))
            # Subtract the maximum entropy value from all entropy weights to avoid numerical instability
            # when applying the exponential function. This keeps values within a manageable range.
            # Then, apply the exponential function to each normalized entropy value, converting them into positive weights.
            # Exponentiating also accentuates differences, making the weighting more pronounced.

            entropy_weights /= entropy_weights.sum()
            # Calculate the sum of all exponential entropy weights.
            # Divide each exponential entropy weight by their total sum to normalize them,
            # ensuring they add up to 1 and can be used as probabilities in a weighted average calculation.

            # Calculate a weighted average of log probabilities across all q values.
            # This step combines the probabilities from different q-gram models, using entropy as weights.
            averaged_log_probabilities = {}
            # Initialize an empty dictionary to store the averaged log probabilities for each letter.

            for letter, log_probs_list in log_probabilities.items():
                if log_probs_list:
                    weighted_log_probs = np.sum([entropy_weights[i] * log_probs
                                                for i, log_probs in enumerate(log_probs_list)], axis=0)
                    # Iterate over each letter and its corresponding list of log probabilities.
                    # Check if the current letter has any calculated log probabilities.
                    # Calculate the weighted sum for the current letter by multiplying each log probability by its corresponding entropy weight.
                    # This assigns more weight to log probabilities from contexts with higher entropy (more informative contexts).
                    # The sum is taken over all q-gram sizes to combine their strengths.
                    averaged_log_probabilities[letter] = weighted_log_probs
                    # Store the weighted sum (averaged log probability) for each letter in the dictionary.
                    # This value represents the combined log probability of the letter being the correct one,
                    # based on all q-gram model predictions and weighted by the entropy of each context.

            # Select the top three letters with the highest probabilities.
            top_three_predictions = heapq.nlargest(3, averaged_log_probabilities.items(), key=lambda item: item[1])
            return [(letter, np.exp(log_prob)) for letter, log_prob in top_three_predictions]
            # Use a heap queue algorithm to find the top three predictions with the highest probabilities.
            # Return these predictions along with their exponential probabilities for interpretation.

    # weights with lower entropy values.
    def predict_missing_letter(self, oov_word):
        # Identify the position of the missing letter (denoted by '_') in the out-of-vocabulary word.
        missing_letter_index = oov_word.index('_')

        # Initialize dictionaries to store the log probabilities of each alphabet letter being the missing one
        # and the entropy weights for each context size (q) in the q-gram models.
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyz'}
        entropy_weights = []

        # Add start (<s>) and end (</s>) boundary markers to the word, to simulate sentence boundaries as in the training data.
        oov_word_with_boundaries = f"<s> {oov_word} </s>"

        # Loop through each q-gram model. The q-gram size determines how many letters (context) are considered around the missing letter.
        for q in self.q_range:
            if q not in self.model:
                continue  # Skip if the model for the current q-gram size is not available.

            model = self.model[q]

            # Determine the number of letters (context size) to consider to the left and right of the missing letter.
            # This context size depends on the position of the missing letter and the size of the q-gram.
            left_size = min(missing_letter_index + 1, q - 1)  # Adjust for <s> at the start.
            right_size = min(len(oov_word) - missing_letter_index, q - 1)  # <s> at the end doesn't affect right size.

            # Extract left and right contexts from the word, including the boundary markers.
            left_context = oov_word_with_boundaries[max(4, missing_letter_index - left_size + 4):missing_letter_index + 4]  # 4 accounts for the length of "<s> ".
            right_context = oov_word_with_boundaries[missing_letter_index + 5:missing_letter_index + 5 + right_size]  # 5 accounts for "<s> " and the '_' character.

            # Remove any extra spaces from the context strings.
            left_context = left_context.strip()
            right_context = right_context.strip()

            # Combine the contexts into single strings, as they would appear in natural text.
            left_context_joined = ' '.join(left_context)
            right_context_joined = ' '.join(right_context)

            # Calculate the entropy of the current context. Entropy here measures the uncertainty of the context.
            # It's used to give more weight to more informative contexts in the final prediction.
            entropy = -sum(model.score(left_context_joined + ' ' + c + ' ' + right_context_joined)
                        for c in 'abcdefghijklmnopqrstuvwxyz')
            entropy_weights.append(entropy)

            # For each possible letter, calculate the log probability of it being the correct one to fill the missing spot.
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                # Create a sequence by inserting each potential letter into the placeholder and score it using the model.
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                log_prob_full = model.score(full_sequence)
                # Store the log probability for this letter and context combination.
                log_probabilities[letter].append(log_prob_full)

            # Invert the entropy weights outside of the letter loop.
            inverted_entropy_weights = np.exp(-np.array(entropy_weights) + np.max(entropy_weights))
            # Normalize the inverted entropy weights
            inverted_entropy_weights /= inverted_entropy_weights.sum()

            # Calculate a weighted average of log probabilities across all q values using the inverted entropy weights.
            averaged_log_probabilities = {}
            for letter, log_probs_list in log_probabilities.items():
                if log_probs_list:
                    weighted_log_probs = np.sum([inverted_entropy_weights[i] * log_probs
                                                for i, log_probs in enumerate(log_probs_list)], axis=0)
                    averaged_log_probabilities[letter] = weighted_log_probs

            # Select the top three letters with the highest probabilities.
            top_three_predictions = heapq.nlargest(3, averaged_log_probabilities.items(), key=lambda item: item[1])
            return [(letter, np.exp(log_prob)) for letter, log_prob in top_three_predictions]
            # Use a heap queue algorithm to find the top three predictions with the highest probabilities.
            # Return these predictions along with their exponential probabilities for interpretation.

    def evaluate_model(self, output_file):
        correct_counts = {1: 0, 2: 0, 3: 0}
        total_words = len(self.test_set)
        top1_valid_predictions = 0
        top2_valid_predictions = 0
        top3_valid_predictions = 0
        predictions = []

        for test_data in self.test_set:
            modified_word, missing_letter, original_word = test_data
            top_three_predictions = self.predict_missing_letter(modified_word)

            correct_found = False
            for rank, (predicted_letter, _) in enumerate(top_three_predictions, start=1):
                reconstructed_word = modified_word.replace('_', predicted_letter)
                if predicted_letter == missing_letter:
                    correct_found = True
                    for i in range(rank, 4):
                        correct_counts[i] += 1  # Update correct counts
                    if rank == 1:
                        top1_valid_predictions += 1
                    if rank <= 2:
                        top2_valid_predictions += 1
                    top3_valid_predictions += 1
                    break  # Stop checking if correct letter is found

                # Check for valid word in TOP1 and TOP2 predictions
                if rank == 1 and reconstructed_word in self.all_words:
                    top1_valid_predictions += 1
                if rank == 2 and reconstructed_word in self.all_words:
                    top2_valid_predictions += 1

            # Check for any valid word in TOP3 predictions if correct letter wasn't found
            if not correct_found:
                if any(modified_word.replace('_', pred_letter) in self.all_words for pred_letter, _ in top_three_predictions):
                    top3_valid_predictions += 1

            predictions.append((modified_word, missing_letter, top_three_predictions))

        top1_recall = top1_valid_predictions / total_words if total_words > 0 else 0.0
        top2_recall = top2_valid_predictions / total_words if total_words > 0 else 0.0
        top3_recall = top3_valid_predictions / total_words if total_words > 0 else 0.0

        self.save_predictions_to_file(correct_counts, top1_recall, top2_recall, top3_recall, total_words, predictions, output_file)

        return correct_counts, top1_recall, top2_recall, top3_recall


    def save_predictions_to_file(self, correct_counts, top1_recall, top2_recall, top3_recall, total_words, predictions, file_name):
        with open(file_name, 'w') as file:
            # Write accuracies and metrics
            file.write(f"TOP1 PRECISION: {correct_counts[1] / total_words:.2%}\n")
            file.write(f"TOP2 PRECISION: {correct_counts[2] / total_words:.2%}\n")
            file.write(f"TOP3 PRECISION: {correct_counts[3] / total_words:.2%}\n")
            file.write(f"TOP1 RECALL: {top1_recall:.2%}\n")
            file.write(f"TOP2 RECALL: {top2_recall:.2%}\n")
            file.write(f"TOP3 RECALL: {top3_recall:.2%}\n\n")

            # Write predictions for each word
            for modified_word, missing_letter, top_three_predictions in predictions:
                top_predicted_letter, top_confidence = top_three_predictions[0]
                file.write(f"Test Word: {modified_word}\nOriginal: {missing_letter}\n")
                file.write(f"Predicted: {top_predicted_letter}\n")
                file.write("Top Three Predictions:\n")
                for rank, (letter, confidence) in enumerate(top_three_predictions, start=1):
                    file.write(f"Rank {rank}: '{letter}' confidence {confidence:.7f}\n")
                file.write("\n")

    def save_set_to_file(self, data_set, file_name):
        with open(file_name, 'w') as file:
            for item in data_set:
                formatted_line = self.format_data_for_saving(item)
                file.write(formatted_line + "\n")

    @staticmethod
    def format_data_for_saving(item):
        if isinstance(item, tuple):
            return f"({item[0]}, {item[1]}, {item[2]})"
        else:
            return f"{item}"

def main():
    logging.info("Starting the main function")

    lm = LanguageModel()
    logging.info("Language model initialized")

    lm.load_corpus()
    logging.info("CMU corpus loaded")

    lm.prepare_datasets()
    logging.info(f"Training set prepared with {len(lm.training_set)} words")
    logging.info(f"Test set prepared with {len(lm.test_set)} words")

    lm.generate_and_load_models()
    logging.info("Q-gram models generated and loaded")

    correct_counts, top1_recall, top2_recall, top3_recall = lm.evaluate_model("cmu_predictions.txt")

    logging.info("Model evaluation completed")
    logging.info(f"TOP1 PRECISION: {correct_counts[1] / len(lm.test_set):.2%}")
    logging.info(f"TOP2 PRECISION: {correct_counts[2] / len(lm.test_set):.2%}")
    logging.info(f"TOP3 PRECISION: {correct_counts[3] / len(lm.test_set):.2%}")
    logging.info(f"TOP1 RECALL: {top1_recall:.2%}")
    logging.info(f"TOP2 RECALL: {top2_recall:.2%}")
    logging.info(f"TOP3 RECALL: {top3_recall:.2%}")

if __name__ == "__main__":
    main()

