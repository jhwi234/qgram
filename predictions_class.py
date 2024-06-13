from collections import defaultdict
import heapq
import numpy as np

class Predictions:
    def __init__(self, model, q_range, unique_characters):
        """
        Initialize the Predictions class with language models, a range of n-gram sizes, and unique characters from the corpus.

        :param model: A dictionary of language models for different n-gram sizes.
        :param q_range: A range of n-gram sizes.
        :param unique_characters: A list of unique characters from the corpus.
        """
        self.model = model
        self.q_range = q_range
        self.unique_characters = unique_characters

    def _extract_contexts(self, test_word, q, missing_letter_index, with_boundaries=True):
        """
        Extracts left and right context around the missing letter.

        :param test_word: The word with the missing letter.
        :param q: The size of the n-gram.
        :param missing_letter_index: The index of the missing letter.
        :param with_boundaries: Whether to include boundary markers.
        :return: A tuple containing the left and right context.
        """
        # Determine the maximum possible context size based on the q-gram model
        max_context_size = q - 1

        # Dynamically calculate the size of the left context based on the position of the missing letter
        left_context_size = min(missing_letter_index, max_context_size)

        # Similarly, calculate the size of the right context, taking into account the word length and position of the missing letter
        right_context_size = min(len(test_word) - missing_letter_index - 1, max_context_size)

        # If with_boundaries is True, include boundary markers at the start and end of the word
        if with_boundaries:
            test_word_with_boundaries = f"<s>{test_word}</s>"  # Adding start <s> and end </s> markers
            adjusted_index = missing_letter_index + 3  # Adjust for the "<s>" at the beginning
            # Extract the context, considering the added boundary markers and adjusted context size
            context_start = max(0, adjusted_index - left_context_size)
            context_end = adjusted_index + right_context_size + 1
            context = test_word_with_boundaries[context_start:context_end]
        else:
            # Extract context without boundary markers, using the dynamically calculated context sizes
            context_start = max(0, missing_letter_index - left_context_size)
            context_end = missing_letter_index + right_context_size + 1
            context = test_word[context_start:context_end]

        return context.strip()

    def _format_sequence(self, left_context, letter, right_context):
        """
        Formats a sequence by combining contexts with a candidate letter.

        :param left_context: The left context.
        :param letter: The candidate letter.
        :param right_context: The right context.
        :return: The formatted sequence.
        """
        return f"{left_context} {letter} {right_context}".strip()

    def _calculate_log_probability(self, model, sequence, bos=True, eos=True):
        """
        Calculates the log probability of a sequence using a specified language model.

        :param model: The language model to use.
        :param sequence: The sequence to score.
        :param bos: Include beginning-of-sequence marker.
        :param eos: Include end-of-sequence marker.
        :return: The log probability of the sequence.
        """
        return model.score(sequence, bos=bos, eos=eos)

    def _select_all_predictions(self, log_probabilities):
        """
        Selects all predictions based on their log probabilities and normalizes them.

        :param log_probabilities: A dictionary of log probabilities for each letter.
        :return: A list of all predictions sorted by normalized probability.
        """
        # Convert log probabilities to actual probabilities
        probabilities = {letter: np.exp(log_prob) for letter, log_prob in log_probabilities.items()}

        # Normalize probabilities
        total_prob = sum(probabilities.values())
        normalized_probabilities = {letter: prob / total_prob for letter, prob in probabilities.items()}

        # Select all predictions sorted by normalized probability
        all_predictions = heapq.nlargest(len(normalized_probabilities), normalized_probabilities.items(), key=lambda item: item[1])
        return all_predictions

    def _predict_missing_letter(self, test_word, missing_letter_index, with_boundaries=True):
        """
        Predicts the missing letter at a specific position using a context-sensitive approach.

        :param test_word: The word with the missing letter.
        :param missing_letter_index: The index of the missing letter.
        :param with_boundaries: Whether to include boundary markers.
        :return: The most probable letter.
        """
        log_probabilities = defaultdict(list)

        # Iterate over the range of q-gram models
        for q in self.q_range:
            model = self.model.get(q)
            if model is None:
                continue  # Skip if the model for the current q is not available

            # Extract the left and right contexts
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=with_boundaries)

            # Calculate log probabilities for each possible letter
            for letter in self.unique_characters:
                sequence = f"{left_context} {letter} {right_context}".strip()
                log_prob = self._calculate_log_probability(model, sequence)
                log_probabilities[letter].append(log_prob)

        # Sum the log probabilities for each letter
        sum_log_probabilities = {letter: np.logaddexp.reduce(log_probs) for letter, log_probs in log_probabilities.items()}
        predictions = self._select_all_predictions(sum_log_probabilities)
        return predictions[0][0]  # Return the most probable letter

    def predict_multiple_missing_letters(self, test_word, with_boundaries=True):
        """
        Predicts multiple missing letters in a word.

        :param test_word: The word with missing letters.
        :param with_boundaries: Whether to include boundary markers.
        :return: The word with all missing letters predicted.
        """
        # Find the indices of all missing letters in the word
        missing_letter_indices = [i for i, char in enumerate(test_word) if char == '_']
        test_word_list = list(test_word)

        # Iterate over each missing letter index and predict the most probable letter
        while missing_letter_indices:
            predictions = {}

            # Collect predictions for all missing indices
            for idx in missing_letter_indices:
                predicted_letter = self._predict_missing_letter(test_word_list, idx, with_boundaries=with_boundaries)
                predictions[idx] = predicted_letter

            # Update the word with the highest confidence prediction
            for idx in sorted(predictions.keys()):
                test_word_list[idx] = predictions[idx]

            # Recalculate missing indices
            missing_letter_indices = [i for i, char in enumerate(test_word_list) if char == '_']

        # Join the list back into a single string and return the predicted word
        return ''.join(test_word_list)

    def context_sensitive(self, test_word):
        """
        Context-sensitive prediction using boundary markers for a single missing letter.

        :param test_word: The word with a single missing letter.
        :return: Predictions sorted by their probabilities.
        """
        missing_letter_index = test_word.index('_')
        log_probabilities = defaultdict(list)

        for q in self.q_range:
            model = self.model.get(q)
            if model is None:
                continue  # Skip if the model for the current q is not available

            # Extract contexts considering boundary markers
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=True)

            # Calculate log probabilities for each possible letter
            for letter in self.unique_characters:
                sequence = f"{left_context} {letter} {right_context}".strip()
                log_prob = self._calculate_log_probability(model, sequence)
                log_probabilities[letter].append(log_prob)

        # Sum the log probabilities for each letter
        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items()}
        return self._select_all_predictions(sum_log_probabilities)

    def context_no_boundary(self, test_word):
        """
        Context prediction without boundary markers. Uses context from both sides of the missing letter, but doesn't consider boundary markers.

        :param test_word: The word with a single missing letter.
        :return: Predictions sorted by their probabilities.
        """
        missing_letter_index = test_word.index('_')
        log_probabilities = defaultdict(list)

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue  # Skip if the model for the current q is not available

            # Extract contexts without considering boundary markers
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=False)

            # Calculate log probabilities for each possible letter
            for letter in self.unique_characters:
                sequence = self._format_sequence(left_context, letter, right_context)
                log_prob = self._calculate_log_probability(model, sequence, bos=False, eos=False)
                log_probabilities[letter].append(log_prob)

        # Sum the log probabilities for each letter
        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items() if log_probs}
        return self._select_all_predictions(sum_log_probabilities)

    def base_prediction(self, test_word):
        """
        Base prediction method. Does not use any context. Does the minimum amount to query the language model correctly.

        :param test_word: The word with a single missing letter.
        :return: Predictions sorted by their probabilities.
        """
        missing_letter_index = test_word.index('_')
        log_probabilities = {}

        # Format the test word to match the training format
        formatted_test_word = " ".join(test_word)

        # Choose the model with the largest n-gram (highest q-value)
        max_q = max(self.q_range)
        model = self.model.get(max_q)
        if not model:
            return []

        # Calculate log probabilities for each possible letter
        for letter in self.unique_characters:
            candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
            log_probability = self._calculate_log_probability(model, candidate_word, bos=False, eos=False)
            log_probabilities[letter] = log_probability

        # Select all letters sorted by their log probabilities
        return self._select_all_predictions(log_probabilities)
