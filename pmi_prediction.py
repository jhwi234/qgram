import heapq
import numpy as np
from nltk.lm import MLE

class PMIPrediction:

    def calculate_letter_frequencies(self, corpus_name):
        # Split the corpus text into words
        words = corpus_name.split(' <w> ')
        words = [word.strip('</w>').strip() for word in words]

        # Count the occurrences of each letter
        letter_counts = {}
        total_letters = 0
        for word in words:
            characters = word.split()
            for char in characters:
                if char.isalpha():  # Only count alphabetical characters
                    letter_counts[char] = letter_counts.get(char, 0) + 1
                    total_letters += 1

        # Calculate the probabilities for each letter
        letter_probabilities = {letter: count / total_letters for letter, count in letter_counts.items()}

        return letter_probabilities


    def pmi_predict_missing_letter(self, corpus_name, oov_word):
        missing_letter_index = oov_word.index('_')
        log_probabilities = {letter: [] for letter in 'abcdefghijklmnopqrstuvwxyzæœ'}
        pmi_weights = []

        boundary_start = '<w> ' if missing_letter_index == 0 else ''
        boundary_end = ' </w>' if missing_letter_index == len(oov_word) - 1 else ''
        oov_word_with_boundaries = f"{boundary_start}{oov_word}{boundary_end}"

        for q in self.q_range:
            if q not in self.models[corpus_name]:
                print(f"No model found for {q}-grams in {corpus_name} corpus.")
                continue
            model = self.models[corpus_name][q]

            # Prepare contexts based on the current q value, ensuring not to exceed bounds
            left_size = min(missing_letter_index, q - 1)
            right_size = min(len(oov_word) - missing_letter_index - 1, q - 1)

            left_context = oov_word_with_boundaries[max(0, missing_letter_index - left_size + len(boundary_start)):missing_letter_index + len(boundary_start)]
            right_context = oov_word_with_boundaries[missing_letter_index + len(boundary_start) + 1:missing_letter_index + len(boundary_start) + 1 + right_size]

            # Ensure there are no extra spaces before or after the context
            left_context = left_context.strip()
            right_context = right_context.strip()

            # Joining contexts with spaces as they would appear in the corpus
            left_context_joined = ' '.join(left_context)
            right_context_joined = ' '.join(right_context)

            # Initialize pmi_weights for this q-gram size
            pmi_weights_for_q = []

            for letter in 'abcdefghijklmnopqrstuvwxyzæœ':
                full_sequence = f"{left_context_joined} {letter} {right_context_joined}".strip()
                p_xy = np.exp(model.score(full_sequence))
                p_x = np.exp(model.score(left_context_joined + ' ' + right_context_joined))
                p_y = 1 / 27  # Assuming uniform distribution for simplicity

                # Compute PMI, guard against log(0) by max with a very small number near zero
                pmi = np.log(max(p_xy / (p_x * p_y), 1e-10))
                pmi = max(pmi, 0)  # Positive PMI only
                pmi_weights_for_q.append(pmi)

                log_probabilities[letter].append(model.score(full_sequence))

            # Normalize the PMI weights for this q
            pmi_weights_for_q = np.array(pmi_weights_for_q)
            pmi_weights_for_q -= np.min(pmi_weights_for_q)
            pmi_weights_for_q = np.exp(pmi_weights_for_q)
            pmi_weights_for_q /= pmi_weights_for_q.sum()

            # Add these PMI weights to the overall list
            pmi_weights.append(pmi_weights_for_q)

        # Normalize across all q values after the main loop
        pmi_weights = np.concatenate(pmi_weights)  # This assumes each sublist has the same length
        pmi_weights -= np.min(pmi_weights)
        pmi_weights = np.exp(pmi_weights)
        pmi_weights /= pmi_weights.sum()

        # Now average the log probabilities across all q values with PPMI weights
        averaged_log_probabilities = {}
        for letter, log_probs_list in log_probabilities.items():
            if log_probs_list:
                # Ensure pmi_weights are matched correctly to log_probs_list
                weighted_log_probs = np.sum(np.array(log_probs_list) * pmi_weights[:len(log_probs_list)])
                averaged_log_probabilities[letter] = weighted_log_probs
                # Trim the used weights
                pmi_weights = pmi_weights[len(log_probs_list):]

        # Apply heapq.nlargest to find the top log probabilities
        top_log_predictions = heapq.nlargest(3, averaged_log_probabilities.items(), key=lambda item: item[1])

        # Convert only the top log probabilities to probabilities
        top_predictions = [(letter, np.exp(log_prob)) for letter, log_prob in top_log_predictions]

        # Return predictions with probabilities
        return top_predictions