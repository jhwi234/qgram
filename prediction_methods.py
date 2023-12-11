from collections import defaultdict
import heapq
import numpy as np

class Predictions:

    alphabet = 'abcdefghijklmnopqrstuvwxyzæœ'

    def __init__(self, model, q_range):
        """Initialize with a set of models and a range of n-gram sizes."""
        self.model = model
        self.q_range = q_range

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

    def _select_top_predictions(self, log_probabilities):
        """Selects the top three predictions based on their log probabilities."""
        top_three = heapq.nlargest(3, log_probabilities.items(), key=lambda item: item[1])
        return [(letter, np.exp(log_prob)) for letter, log_prob in top_three]

    def context_sensitive(self, test_word):
        """Prediction method that considers different q-gram contexts."""
        missing_letter_index = test_word.index('_')
        log_probabilities = defaultdict(list)

        for q in self.q_range:
            model = self.model.get(q)
            if model is None:
                continue

            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=True)
            for letter in self.alphabet:
                sequence = f"{left_context} {letter} {right_context}".strip()
                log_prob = model.score(sequence)
                log_probabilities[letter].append(log_prob)

        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items()}
        return self._select_top_predictions(sum_log_probabilities)

    def context_no_boundary(self, test_word):
        """Prediction method determines context size without boundary markers."""
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in self.alphabet}

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Extract contexts without boundary markers
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=False)

            for letter in self.alphabet:
                sequence = self._format_sequence(left_context, letter, right_context)
                log_prob = self._calculate_log_probability(model, sequence, bos=False, eos=False)
                log_probabilities[letter].append(log_prob)

        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items() if log_probs}
        return self._select_top_predictions(sum_log_probabilities)

    def context_insensitive(self, test_word):
        """Prediction method that ignores context size and boundary markers."""
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in self.alphabet}

        # Format the test word to match the training format (with spaces between characters).
        formatted_test_word = " ".join(test_word)

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            for letter in self.alphabet:
                # Form the candidate word by replacing the underscore with the letter and adding spaces.
                candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
                
                # Use helper method for log probability calculation.
                log_probability = self._calculate_log_probability(model, candidate_word, bos=False, eos=False)
                log_probabilities[letter].append(log_probability)

        # Sum log probabilities across all q values and select top three predictions.
        sum_log_probabilities = {letter: sum(log_probs) for letter, log_probs in log_probabilities.items() if log_probs}
        return self._select_top_predictions(sum_log_probabilities)

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

        for letter in self.alphabet:
            # Form the candidate word by replacing the underscore with the letter and adding spaces.
            candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
            
            # Use helper method for log probability calculation.
            log_probability = self._calculate_log_probability(model, candidate_word, bos=False, eos=False)
            log_probabilities[letter] = log_probability

        # Select the top three letters with the highest log probabilities.
        return self._select_top_predictions(log_probabilities)
    
    def entropy_weighted(self, test_word):
        """Context sensitive prediction method using entropy weighting."""
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in self.alphabet}
        entropy_weights = []

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Extract contexts using helper method
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=True)

            # Calculate entropy for the current context
            entropy = self._calculate_entropy(model, left_context, right_context)
            entropy_weights.append(entropy)

            # Calculate log probabilities for each letter
            for letter in self.alphabet:
                full_sequence = self._format_sequence(left_context, letter, right_context)
                log_prob_full = self._calculate_log_probability(model, full_sequence)
                log_probabilities[letter].append(log_prob_full)

        # Normalize entropy weights
        entropy_weights = self._normalize_entropy_weights(entropy_weights)

        # Average the log probabilities across all q values with entropy weights
        averaged_log_probabilities = self._apply_entropy_weights(log_probabilities, entropy_weights)

        # Select top three predictions
        return self._select_top_predictions(averaged_log_probabilities)

    def interpolation_weighted(self, test_word):
        """Context sensitive prediction method using interpolation weighting."""
        missing_letter_index = test_word.index('_')
        probabilities = {letter: [] for letter in self.alphabet}
        lambda_weights = self.calculate_lambda_weights()

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries=True)

            # Probability calculation using list comprehension
            for letter in self.alphabet:
                full_sequence = self._format_sequence(left_context, letter, right_context)
                prob_full = np.exp(self._calculate_log_probability(model, full_sequence))  # Convert log probability to linear probability
                probabilities[letter].append(prob_full * lambda_weights[q])

        # Interpolation using dictionary comprehension
        interpolated_probabilities = {
            letter: sum(probs_list)
            for letter, probs_list in probabilities.items() if probs_list
        }

        # Efficient selection of top three predictions
        return self._select_top_predictions(interpolated_probabilities)

    # Supporting methods for entropy-weighted
    def _calculate_entropy(self, model, left_context, right_context):
        """Calculates the entropy for the current context."""
        return -sum(np.exp(model.score(f"{left_context} {c} {right_context}")) for c in self.alphabet)

    def _normalize_entropy_weights(self, entropy_weights):
        """Normalizes entropy weights."""
        entropy_weights = np.exp(entropy_weights - np.max(entropy_weights))
        entropy_weights /= entropy_weights.sum()
        return entropy_weights

    def _apply_entropy_weights(self, log_probabilities, entropy_weights):
        """Applies entropy weights to log probabilities."""
        averaged_log_probabilities = {}
        for letter, log_probs_list in log_probabilities.items():
            if log_probs_list:
                weighted_log_probs = np.sum([entropy_weights[i] * log_probs
                                            for i, log_probs in enumerate(log_probs_list)], axis=0)
                averaged_log_probabilities[letter] = weighted_log_probs
        return averaged_log_probabilities

    def _calculate_lambda_weights(self):
        """Calculates lambda weights for each n-gram size."""
        lambda_weights = {q: 1.0 / len(self.q_range) for q in self.q_range}
        return lambda_weights