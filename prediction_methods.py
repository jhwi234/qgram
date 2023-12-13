from collections import defaultdict
import heapq
import numpy as np

class Predictions:
    def __init__(self, model, q_range, unique_characters):
        """Initialize with a set of models, a range of n-gram sizes, and unique characters."""
        self.model = model
        self.q_range = q_range
        self.unique_characters = unique_characters

    def _extract_contexts(self, test_word, q, missing_letter_index, with_boundaries=True):
        """Extracts left and right contexts around the missing letter in the word."""
        left_size = min(missing_letter_index, q - 1)
        right_size = min(len(test_word) - missing_letter_index - 1, q - 1)

        if with_boundaries:
            test_word = f"<s> {test_word} </s>"
            left_context = test_word[max(4, missing_letter_index - left_size + 4):missing_letter_index + 4]
            right_context = test_word[missing_letter_index + 5:missing_letter_index + 5 + right_size]
        else:
            left_context = test_word[:missing_letter_index][-left_size:]
            right_context = test_word[missing_letter_index + 1:][:right_size]

        return ' '.join(left_context.strip()), ' '.join(right_context.strip())

    def _format_sequence(self, left_context, letter, right_context):
        """Formats a sequence by joining the left context, a candidate letter, and the right context."""
        return f"{left_context} {letter} {right_context}".strip()

    def _calculate_log_probability(self, model, sequence, bos=True, eos=True):
        """Calculates the log probability of a sequence using a specified model."""
        return model.score(sequence, bos=bos, eos=eos)
    
    def _select_all_predictions(self, log_probabilities):
        """Selects all predictions based on their log probabilities."""
        all_predictions = heapq.nlargest(len(log_probabilities), log_probabilities.items(), key=lambda item: item[1])
        return [(letter, np.exp(log_prob)) for letter, log_prob in all_predictions]

    def context_sensitive(self, test_word):
        """Prediction method that determines the context size with boundary markers."""
        missing_letter_index = test_word.index('_')
        log_probabilities = defaultdict(list)

        for q in self.q_range:
            model = self.model.get(q)
            if model is None:
                continue
            # Extract contexts with boundary markers
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=True)
            for letter in self.unique_characters:
                sequence = f"{left_context} {letter} {right_context}".strip()
                log_prob = model.score(sequence)
                log_probabilities[letter].append(log_prob)

        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items()}
        return self._select_all_predictions(sum_log_probabilities)

    def context_no_boundary(self, test_word):
        """Prediction method determines context size without boundary markers."""
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in self.unique_characters}

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Extract contexts without boundary markers
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=False)

            for letter in self.unique_characters:
                sequence = self._format_sequence(left_context, letter, right_context)
                log_prob = self._calculate_log_probability(model, sequence, bos=False, eos=False)
                log_probabilities[letter].append(log_prob)

        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items() if log_probs}
        return self._select_all_predictions(sum_log_probabilities)

    def context_insensitive(self, test_word):
        """Prediction method that ignores context size and boundary markers."""
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in self.unique_characters}

        # Format the test word to match the training format (with spaces between characters).
        formatted_test_word = " ".join(test_word)

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            for letter in self.unique_characters:
                # Form the candidate word by replacing the underscore with the letter and adding spaces.
                candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
                
                # Use helper method for log probability calculation.
                log_probability = self._calculate_log_probability(model, candidate_word, bos=False, eos=False)
                log_probabilities[letter].append(log_probability)

        # Sum log probabilities across all q values and select top three predictions.
        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items() if log_probs}
        return self._select_all_predictions(sum_log_probabilities)

    def base_prediction(self, test_word):
        """Base prediction method using the highest q-value model."""
        missing_letter_index = test_word.index('_')
        log_probabilities = {}

        # Format the test word to match the training format (with spaces between characters).
        formatted_test_word = " ".join(test_word)

        # Choose the model with the largest n-gram.
        max_q = max(self.q_range)
        model = self.model.get(max_q)
        if not model:
            return []

        for letter in self.unique_characters:
            # Form the candidate word by replacing the underscore with the letter and adding spaces.
            candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
            
            # Use helper method for log probability calculation.
            log_probability = self._calculate_log_probability(model, candidate_word, bos=False, eos=False)
            log_probabilities[letter] = log_probability

        # Select the top three letters with the highest log probabilities.
        return self._select_all_predictions(log_probabilities)