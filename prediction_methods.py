import heapq
import numpy as np

class Predictions:
    def __init__(self, model, q_range):
        self.model = model
        self.q_range = q_range

    # version that averages the log probabilities across all q values with entropy weighting and determines context size based on the missing letter index uses boundary markers and interpolation
    def entropy_weighted_prediction(self, test_word):
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}
        entropy_weights = []
        test_word_with_boundaries = f"<s> {test_word} </s>"

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Determine context size.
            left_size = min(missing_letter_index, q - 1)
            right_size = min(len(test_word) - missing_letter_index - 1, q - 1)

            # Extract left and right contexts from the word, including the boundary markers.
            left_context = test_word_with_boundaries[max(4, missing_letter_index - left_size + 4):missing_letter_index + 4]
            right_context = test_word_with_boundaries[missing_letter_index + 5:missing_letter_index + 5 + right_size]

            # Combine the contexts into single strings.
            left_context_joined = ' '.join(left_context.strip())
            right_context_joined = ' '.join(right_context.strip())

            # Calculate entropy for the current context
            entropy = -sum(model.score(left_context_joined + ' ' + c + ' ' + right_context_joined)
                        for c in 'abcdefghijklmnopqrstuvwxyzæœ')
            entropy_weights.append(entropy)

            for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                log_prob_full = model.score(full_sequence)
                log_probabilities[letter].append(log_prob_full)

            # Normalize entropy weights
            entropy_weights = np.exp(entropy_weights - np.max(entropy_weights))
            entropy_weights /= entropy_weights.sum()

            # Average the log probabilities across all q values with entropy weights
            averaged_log_probabilities = {}
            for letter, log_probs_list in log_probabilities.items():
                if log_probs_list:
                    weighted_log_probs = np.sum([entropy_weights[i] * log_probs
                                                for i, log_probs in enumerate(log_probs_list)], axis=0)
                    averaged_log_probabilities[letter] = weighted_log_probs

        top_log_predictions = heapq.nlargest(3, averaged_log_probabilities.items(), key=lambda item: item[1])
        top_predictions = [(letter, np.exp(log_prob)) for letter, log_prob in top_log_predictions]
        return top_predictions
    
    # version that sums the log probabilities across all q values and multiplies them with weights using interpolation and determines context size based on the missing letter index uses boundary markers and interpolation
    def interpolation_weighted_prediction(self, test_word):
        missing_letter_index = test_word.index('_')
        probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}
        test_word_with_boundaries = f"<s> {test_word} </s>"
        lambda_weights = self.calculate_lambda_weights()

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Determine context size.
            left_size = min(missing_letter_index + 1, q - 1)
            right_size = min(len(test_word) - missing_letter_index, q - 1)

            # Extract left and right contexts from the word, including the boundary markers.
            left_context = test_word_with_boundaries[max(4, missing_letter_index - left_size + 4):missing_letter_index + 4]
            right_context = test_word_with_boundaries[missing_letter_index + 5:missing_letter_index + 5 + right_size]

            left_context = left_context.strip()
            right_context = right_context.strip()

            # Combine the contexts into single strings.
            left_context_joined = ' '.join(left_context)
            right_context_joined = ' '.join(right_context)

            # Probability calculation using list comprehension
            for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                prob_full = np.exp(model.score(full_sequence))  # Convert log probability to linear probability
                probabilities[letter].append(prob_full * lambda_weights[q])

        # Interpolation using dictionary comprehension
        interpolated_probabilities = {
            letter: sum(probs_list)
            for letter, probs_list in probabilities.items() if probs_list
        }

        # Efficient selection of top three predictions
        return heapq.nlargest(3, interpolated_probabilities.items(), key=lambda item: item[1])

    def calculate_lambda_weights(self):
        # Define lambda weights for each n-gram size. These should sum up to 1.
        lambda_weights = {
            1: 0.0,  # Unigram
            2: 0.2,  # Bigram
            3: 0.2,  # Trigram
            4: 0.2,
            5: 0.2,
            6: 0.2
        }
        return lambda_weights

    # version that sums the log probabilities across all q values and determines context size based on the missing letter index uses boundary markers   
    def context_sensitive_prediction(self, test_word):
        missing_letter_index = test_word.index('_')
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}
        test_word_with_boundaries = f"<s> {test_word} </s>"

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Determine context size.
            left_size = min(missing_letter_index + 1, q - 1)
            right_size = min(len(test_word) - missing_letter_index, q - 1)

            # Extract left and right contexts from the word, including the boundary markers.
            left_context = test_word_with_boundaries[max(4, missing_letter_index - left_size + 4):missing_letter_index + 4]
            right_context = test_word_with_boundaries[missing_letter_index + 5:missing_letter_index + 5 + right_size]

            left_context = left_context.strip()
            right_context = right_context.strip()

            # Combine the contexts into single strings.
            left_context_joined = ' '.join(left_context)
            right_context_joined = ' '.join(right_context)

            # Calculate log probability for each letter
            for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                log_prob_full = model.score(full_sequence)
                log_probabilities[letter].append(log_prob_full)

        # Sum log probabilities across all q values
        sum_log_probabilities = {
            letter: sum(log_probs_list)
            for letter, log_probs_list in log_probabilities.items() if log_probs_list
        }

        # Select top three predictions
        top_three_predictions = heapq.nlargest(3, sum_log_probabilities.items(), key=lambda item: item[1])
        return [(letter, np.exp(log_prob)) for letter, log_prob in top_three_predictions]

    # version that determines context size based on the missing letter index. does not use boundary markers.
    def context_no_boundary_prediction(self, test_word):
        # Identify the position of the missing letter (denoted by '_').
        missing_letter_index = test_word.index('_')

        # Initialize a dictionary to store the log probabilities of each alphabet letter being the missing one.
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Determine context size.
            left_size = min(missing_letter_index, q - 1)
            right_size = min(len(test_word) - missing_letter_index - 1, q - 1)

            # Extract left and right contexts from the word.
            left_context = test_word[:missing_letter_index][-left_size:]
            right_context = test_word[missing_letter_index + 1:][:right_size]

            # Combine the contexts into single strings.
            left_context_joined = ' '.join(left_context)
            right_context_joined = ' '.join(right_context)

            # Calculate log probability for each letter.
            for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                log_prob_full = model.score(full_sequence, bos=False, eos=False)
                log_probabilities[letter].append(log_prob_full)

        # Sum log probabilities across all q values.
        sum_log_probabilities = {
            letter: sum(log_probs_list)
            for letter, log_probs_list in log_probabilities.items() if log_probs_list
        }

        # Select the top three letters with the highest probabilities.
        top_three_predictions = heapq.nlargest(3, sum_log_probabilities.items(), key=lambda item: item[1])
        return [(letter, np.exp(log_prob)) for letter, log_prob in top_three_predictions]

    # version that uses no boundary markers or context determination but sums the log probabilities across all q values
    def context_free_prediction(self, test_word):
        # Identify the position of the missing letter (denoted by '_').
        missing_letter_index = test_word.index('_')

        # Initialize a dictionary to store the log probabilities.
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}

        # Format the test word to match the training format (with spaces between characters).
        formatted_test_word = " ".join(test_word)

        for q in self.q_range:
            model = self.model.get(q)
            if not model:
                continue

            # Generate candidate words for each letter in the alphabet.
            for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
                # Form the candidate word by replacing the underscore with the letter and adding spaces.
                candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
                
                # Score the candidate word using the KenLM model.
                log_probability = model.score(candidate_word, bos=False, eos=False)
                
                # Store the log probability.
                log_probabilities[letter].append(log_probability)

        # Sum log probabilities across all q values.
        sum_log_probabilities = {
            letter: sum(log_probs_list)
            for letter, log_probs_list in log_probabilities.items() if log_probs_list
        }

        # Select the top three letters with the highest probabilities.
        top_three_predictions = heapq.nlargest(3, sum_log_probabilities.items(), key=lambda item: item[1])
        return [(letter, np.exp(log_prob)) for letter, log_prob in top_three_predictions]

    # Base form of predict_missing_letter method only formats and scores the candidate words uses highest q value
    def base_prediction(self, test_word):
        # Identify the position of the missing letter (denoted by '_').
        missing_letter_index = test_word.index('_')

        # Initialize a dictionary to store the log probabilities.
        log_probabilities = {}

        # Format the test word to match the training format (with spaces between characters).
        formatted_test_word = " ".join(test_word)

        # Choose the model with the largest n-gram (assuming larger n-grams are better).
        # This removes the need to loop over different q values.
        max_q = max(self.q_range)
        model = self.model.get(max_q)
        if not model:
            return []

        # Generate candidate words for each letter in the alphabet and score them.
        for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
            candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
            log_probability = model.score(candidate_word, bos=False, eos=False)
            log_probabilities[letter] = log_probability

        # Select the top three letters with the highest log probabilities.
        top_three_predictions = heapq.nlargest(3, log_probabilities.items(), key=lambda item: item[1])
        return [(letter, np.exp(log_prob)) for letter, log_prob in top_three_predictions]

