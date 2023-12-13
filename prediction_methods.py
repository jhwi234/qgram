from collections import defaultdict
import heapq
import numpy as np

class Predictions:
    def __init__(self, model, q_range, unique_characters):
        # Initialize with language models, a range of n-gram sizes, and unique characters from the corpus
        self.model = model
        self.q_range = q_range
        self.unique_characters = unique_characters

    # Extracts left and right context around the missing letter
    def _extract_contexts(self, test_word, q, missing_letter_index, with_boundaries=True):
        # Calculate the size of left and right contexts based on the position of the missing letter
        left_size = min(missing_letter_index, q - 1)
        right_size = min(len(test_word) - missing_letter_index - 1, q - 1)

        # With boundary markers, add start and end symbols to the word
        if with_boundaries:
            test_word = f"<s> {test_word} </s>"
            # Extract contexts considering boundary markers
            left_context = test_word[max(4, missing_letter_index - left_size + 4):missing_letter_index + 4]
            right_context = test_word[missing_letter_index + 5:missing_letter_index + 5 + right_size]
        else:
            # Extract contexts without boundary markers
            left_context = test_word[:missing_letter_index][-left_size:]
            right_context = test_word[missing_letter_index + 1:][:right_size]

        return ' '.join(left_context.strip()), ' '.join(right_context.strip())

    # Formats a sequence by combining contexts with a candidate letter
    def _format_sequence(self, left_context, letter, right_context):
        # Concatenate left context, letter, and right context into a single string
        return f"{left_context} {letter} {right_context}".strip()

    # Calculates the log probability of a sequence using a specified language model
    def _calculate_log_probability(self, model, sequence, bos=True, eos=True):
        # Compute the log probability of the sequence with the given model
        return model.score(sequence, bos=bos, eos=eos)

    # Selects all predictions based on their log probabilities
    def _select_all_predictions(self, log_probabilities):
        # Select all predictions sorted by log probability, converting log to actual probabilities
        all_predictions = heapq.nlargest(len(log_probabilities), log_probabilities.items(), key=lambda item: item[1])
        return [(letter, np.exp(log_prob)) for letter, log_prob in all_predictions]

    # Context-sensitive prediction using boundary markers
    def context_sensitive(self, test_word):
        # Find the index of the missing letter
        missing_letter_index = test_word.index('_')
        log_probabilities = defaultdict(list)

        for q in self.q_range:
            model = self.model.get(q)
            # Skip if the model for the current q is not available
            if model is None:
                continue

            # Extract contexts considering boundary markers
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=True)

            for letter in self.unique_characters:
                # Create sequences for each letter and calculate their probabilities
                sequence = f"{left_context} {letter} {right_context}".strip()
                log_prob = model.score(sequence)
                log_probabilities[letter].append(log_prob)

        # Sum log probabilities for each letter across different q models
        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items()}
        return self._select_all_predictions(sum_log_probabilities)

    # Context prediction without boundary markers. Uses context from both sides of the missing letter, but doesn't consider boundary markers.
    def context_no_boundary(self, test_word):
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in self.unique_characters}

        for q in self.q_range:
            model = self.model.get(q)
            # Skip if the model for the current q is not available
            if not model:
                continue

            # Extract contexts without considering boundary markers
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=False)

            for letter in self.unique_characters:
                # Create sequences for each letter and calculate their probabilities
                sequence = self._format_sequence(left_context, letter, right_context)
                log_prob = self._calculate_log_probability(model, sequence, bos=False, eos=False)
                log_probabilities[letter].append(log_prob)

        # Sum log probabilities for each letter across different q models
        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items() if log_probs}
        return self._select_all_predictions(sum_log_probabilities)

    # Base prediction method. Does not use any context. Does the minimum amount to query the language model correctly.
    def base_prediction(self, test_word):
        missing_letter_index = test_word.index('_')
        log_probabilities = {}

        # Format the test word to match the training format
        formatted_test_word = " ".join(test_word)

        # Choose the model with the largest n-gram (highest q-value)
        max_q = max(self.q_range)
        model = self.model.get(max_q)
        if not model:
            return []

        for letter in self.unique_characters:
            # Replace underscore with each letter and calculate probabilities
            candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
            log_probability = self._calculate_log_probability(model, candidate_word, bos=False, eos=False)
            log_probabilities[letter] = log_probability

        # Select all letters sorted by their log probabilities
        return self._select_all_predictions(log_probabilities)