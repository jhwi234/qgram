<<<<<<< HEAD
import logging
import csv
=======
import csv
import logging
import math
>>>>>>> 9698c3277e395c0ecb9e118b3e05e3169f439863
from predictions_class import Predictions

class EvaluateModel:
    def __init__(self, corpus_manager, split_type=None, log_initialization_details=True):
        # Initialization and dataset preparation using the provided corpus_manager
        self.corpus_manager = corpus_manager
        self.corpus_name = corpus_manager.corpus_name 
        self.config = corpus_manager.config
        self.model = corpus_manager.model  # Loaded language models
        self.corpus = corpus_manager.corpus
        self.train_set = corpus_manager.train_set
        self.test_set = corpus_manager.test_set
        self.all_words = corpus_manager.all_words
        self.split_type = split_type 

        # Extract unique characters from the corpus
        unique_characters = corpus_manager.extract_unique_characters()
        self.unique_character_count = len(unique_characters)

        # Initialize counts for calculating recall and precision
        self.actual_missing_letter_occurrences = {char: 0 for char in unique_characters}
        self.correct_top_predictions = {char: 0 for char in unique_characters}
        self.top_predicted_counts = {char: 0 for char in unique_characters}

        # Initialize prediction class
        self.q_range = range(self.config.q_range[0], self.config.q_range[1] + 1)
        self.predictor = Predictions(self.model, self.q_range, unique_characters)

        # Logging model initialization details
        if log_initialization_details:
            self.log_initialization_details()

        # Retrieve the appropriate prediction method
        prediction_methods = {
            'context_sensitive': self.predictor.context_sensitive,
            'context_no_boundary': self.predictor.context_no_boundary,
            'base_prediction': self.predictor.base_prediction
        }
        self.prediction_method = prediction_methods.get(self.config.prediction_method_name, self.predictor.context_sensitive)

    def log_initialization_details(self):
        # Logging model and configuration details
        logging.info(f'Language Model for {self.corpus_name} initialized with:')
        logging.info(f'Seed: {self.config.seed}')
        logging.info(f'Q-gram Range: {self.config.q_range}')
        logging.info(f'Train-Test Split Configuration: {self.config.split_config}')
        logging.info(f'Training Set Size: {len(self.train_set)}')
        logging.info(f'Testing Set Size: {len(self.test_set)}')
        logging.info(f'Vowel Replacement Ratio: {self.config.vowel_replacement_ratio}')
        logging.info(f'Consonant Replacement Ratio: {self.config.consonant_replacement_ratio}')
        logging.info(f'Unique Character Count: {self.unique_character_count}')
        logging.info(f'Minimum Word Length: {self.config.min_word_length}')

    def compute_accuracy(self, predictions) -> dict:
        # Initialize a dictionary to track accuracy for three ranks: TOP1, TOP2, and TOP3.
        accuracy_counts = {1: 0, 2: 0, 3: 0}  # Ensuring all ranks are initialized.
        total_test_words = len(self.test_set)  # Total number of words in the test set.

        for _, missing_letter, _, all_predictions, _ in predictions:
            # Identify the highest rank (1, 2, or 3) where the correct prediction (missing_letter) is made.
            correct_rank = next((rank for rank, (predicted_letter, _) in enumerate(all_predictions, start=1) if predicted_letter == missing_letter), None)
            
            # If a correct prediction is made, increment accuracy counts for that rank and all higher ranks.
            if correct_rank:
                for rank in range(correct_rank, 4):  # Loop from correct rank to 3.
                    accuracy_counts[rank] += 1  # Increment the count for each relevant rank.

        # Calculate total accuracy for each rank (1, 2, 3) by dividing the number of accurate predictions by total test words.
        total_accuracy = {k: accuracy_counts[k] / total_test_words for k in accuracy_counts}
        return total_accuracy

    def compute_validity(self, predictions) -> dict:
        # Initialize a dictionary to track validity for three ranks: TOP1, TOP2, and TOP3.
        validity_counts = {1: 0, 2: 0, 3: 0}
        total_test_words = len(self.test_set)  # Total number of words in the test set.

        for modified_word, _, _, all_predictions, _ in predictions:
            valid_word_found = False  # Flag to indicate if a valid word has been found.
            for rank, (predicted_letter, _) in enumerate(all_predictions, start=1):
                # If a valid word hasn't been found yet, check the current prediction.
                if not valid_word_found:
                    # Reconstruct the word by replacing the underscore with the predicted letter.
                    reconstructed_word = modified_word.replace('_', predicted_letter)
                    # Check if the reconstructed word exists in the corpus (valid word).
                    if reconstructed_word in self.all_words:
                        # If valid, increment validity counts for that rank and all higher ranks.
                        for i in range(rank, 4):  # Loop from current rank to 3.
                            validity_counts[i] += 1
                        valid_word_found = True  # Update flag since a valid word is found.

        # Calculate total validity for each rank (1, 2, 3) by dividing the number of valid predictions by total test words.
        total_validity = {k: validity_counts[k] / total_test_words for k in validity_counts}
        return total_validity

    def compute_metric(self, correct_counts, total_counts) -> dict:
        # Generalized method to calculate metrics like recall and precision.
        # 'correct_counts' is a dictionary of how many times each character was correctly predicted.
        # 'total_counts' is a dictionary of the total occurrences for recall or total predictions for precision.
        # The metric is calculated as a ratio of correct counts to total counts for each character.
        return {
            char: (correct_counts[char] / total_counts[char] if total_counts[char] > 0 else 0)
            for char in correct_counts
        }

    def evaluate_character_predictions(self, prediction_method) -> tuple[dict, list]:
        predictions = []

        # Iterate over each word in the test set to evaluate predictions.
        for modified_word, target_letter, original_word in self.test_set:
            # Increment the count of occurrences of the target letter for recall calculation.
            self.actual_missing_letter_occurrences[target_letter] += 1

            # Get predictions for the modified word using the provided prediction method.
            all_predictions = prediction_method(modified_word)
            if all_predictions:
                # Increment count of predictions for precision calculation.
                top_predicted_char = all_predictions[0][0]
                self.top_predicted_counts[top_predicted_char] += 1

                # If the top prediction matches the target letter, increment the correct prediction count.
                if top_predicted_char == target_letter:
                    self.correct_top_predictions[target_letter] += 1

            # Determine the rank at which the correct letter is predicted, if at all.
            correct_letter_rank = next((rank for rank, (retrieved_letter, _) in enumerate(all_predictions, start=1) 
                                        if retrieved_letter == target_letter), None)
            # Append the detailed prediction information for each test word.
            predictions.append((modified_word, target_letter, original_word, all_predictions[:3], correct_letter_rank))

        # Compute various evaluation metrics including accuracy, validity, recall, and precision.
        accuracy_metrics = self.compute_accuracy(predictions)
        validity_metrics = self.compute_validity(predictions)
        recall_metrics = self.compute_metric(self.correct_top_predictions, self.actual_missing_letter_occurrences)
        precision_metrics = self.compute_metric(self.correct_top_predictions, self.top_predicted_counts)

        # Return a comprehensive dictionary of all computed metrics.
        return {'accuracy': accuracy_metrics, 'validity': validity_metrics, 
                'recall': recall_metrics, 'precision': precision_metrics, 
                'total_words': len(self.test_set)}, predictions
<<<<<<< HEAD

=======
    
>>>>>>> 9698c3277e395c0ecb9e118b3e05e3169f439863
    def save_recall_precision_stats(self, evaluation_metrics):
        # Retrieve recall and precision metrics from the evaluation_metrics dictionary
        recall_metrics = evaluation_metrics['recall']
        precision_metrics = evaluation_metrics['precision']

<<<<<<< HEAD
        # Sort metrics by Total Relevant (Actual Missing Letter Occurrences) in descending order
        sorted_metrics = sorted(
            [
                (char, 
                 self.actual_missing_letter_occurrences[char], 
                 self.correct_top_predictions[char], 
                 self.top_predicted_counts[char],
                 recall_metrics[char], 
                 precision_metrics[char]
                ) for char in recall_metrics
            ], 
            key=lambda item: item[1], reverse=True
        )

        # Save sorted metrics to a file
        metrics_file_path = self.config.csv_dir / f'{self.corpus_name}_recall_precision_metrics.csv'
        with metrics_file_path.open('w', encoding='utf-8') as file:
            file.write('Character, Total_Missing_Letter_Occurrences, Total_Correctly_Retrieved, Total_Predictions, Recall, Precision\n')
            for char, total_relevant, correctly_retrieved, total_predictions, recall, precision in sorted_metrics:
                file.write(f'{char}, {total_relevant}, {correctly_retrieved}, {total_predictions}, {recall:.4f}, {precision:.4f}\n')
                
    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
        # Check if the split type is provided and adjust the filename accordingly
=======
        # Total occurrences of all letters
        total_letters = sum(self.actual_missing_letter_occurrences.values())

        # Compute F1-score and G-Mean for each character
        f1_scores = {}
        g_means = {}
        weighted_f1_scores = {}
        weighted_g_means = {}
        for char in recall_metrics:
            precision = precision_metrics[char]
            recall = recall_metrics[char]
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            g_mean = math.sqrt(precision * recall)
            
            f1_scores[char] = f1
            g_means[char] = g_mean
            
            # Calculate weighted scores considering the total occurrences of each letter as a weight
            weight = self.actual_missing_letter_occurrences[char] / total_letters
            weighted_f1_scores[char] = f1 * weight
            weighted_g_means[char] = g_mean * weight

        # Sorting metrics by weighted F1-score in descending order might not be meaningful since each character's weighted score is influenced by its frequency, 
        # but we'll sort by the original F1-Score for consistency in the output
        sorted_metrics = sorted(
            [(char, self.actual_missing_letter_occurrences[char], self.correct_top_predictions[char], self.top_predicted_counts[char], recall_metrics[char], precision_metrics[char], f1_scores[char], g_means[char], weighted_f1_scores[char], weighted_g_means[char]) for char in recall_metrics],
            key=lambda item: item[6], reverse=True  # Sort by F1-score
        )

        # Save sorted metrics to a file, including the new weighted columns
        metrics_file_path = self.config.csv_dir / f'{self.corpus_name}_recall_precision_f1_gmean_weighted.csv'
        with metrics_file_path.open('w', encoding='utf-8') as file:
            file.write('Character,Total_Missing_Letter_Occurrences,Total_Correctly_Retrieved,Total_Predictions,Recall,Precision,F1-Score,G-Mean,Weighted F1,Weighted G-Mean\n')
            for char, total_relevant, correctly_retrieved, total_predictions, recall, precision, f1_score, g_mean, weighted_f1, weighted_g_mean in sorted_metrics:
                file.write(f'{char},{total_relevant},{correctly_retrieved},{total_predictions},{recall:.4f},{precision:.4f},{f1_score:.4f},{g_mean:.4f},{weighted_f1:.4f},{weighted_g_mean:.4f}\n')
    
    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
>>>>>>> 9698c3277e395c0ecb9e118b3e05e3169f439863
        split_type_str = f"_{self.split_type}" if self.split_type else ""
        csv_file_path = self.config.csv_dir / (
            f'{self.corpus_name}_{prediction_method_name}{split_type_str}_split'
            f'{self.config.split_config}_qrange{self.config.q_range[0]}-'
            f'{self.config.q_range[1]}_prediction.csv'
        )
        with csv_file_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
<<<<<<< HEAD
            # Adjusted columns to include top three predictions
=======
>>>>>>> 9698c3277e395c0ecb9e118b3e05e3169f439863
            writer.writerow([
                'Tested_Word', 'Original_Word', 'Correct_Letter', 
                'Top1_Predicted_Letter', 'Top1_Confidence', 'Top1_Is_Valid', 'Top1_Is_Accurate',
                'Top2_Predicted_Letter', 'Top2_Confidence', 'Top2_Is_Valid', 'Top2_Is_Accurate',
                'Top3_Predicted_Letter', 'Top3_Confidence', 'Top3_Is_Valid', 'Top3_Is_Accurate',
<<<<<<< HEAD
                'Correct_Letter_Rank'
            ])

            for mod_word, miss_letter, orig_word, top_preds, cor_letter_rank in predictions:
                row = [mod_word, orig_word, miss_letter]

                # Process each of the top three predictions
=======
                'Correct_Letter_Rank', 'In_Training_Set'
            ])

            # Adjusted to directly convert self.train_set to a set without unpacking
            training_words_set = set(self.train_set)
            for mod_word, miss_letter, orig_word, top_preds, cor_letter_rank in predictions:
                row = [mod_word, orig_word, miss_letter]

>>>>>>> 9698c3277e395c0ecb9e118b3e05e3169f439863
                for predicted_letter, confidence in top_preds:
                    reconstructed_word = mod_word.replace('_', predicted_letter)
                    is_valid = 1 if reconstructed_word in self.all_words else 0
                    is_accurate = 1 if predicted_letter == miss_letter else 0

<<<<<<< HEAD
                    # Append prediction details to the row
                    row.extend([predicted_letter, confidence, is_valid, is_accurate])

                # Append correct letter rank to the row
                row.append(cor_letter_rank)

                # Write the complete row to the CSV
=======
                    row.extend([predicted_letter, confidence, is_valid, is_accurate])

                row.append(cor_letter_rank)
                row.append(1 if orig_word in training_words_set else 0)  # Check if in training set

>>>>>>> 9698c3277e395c0ecb9e118b3e05e3169f439863
                writer.writerow(row)

    def save_summary_stats_txt(self, evaluation_metrics, predictions, prediction_method_name):
        # File path for saving prediction summary
        output_file_path = self.config.text_dir / f'{self.corpus_name}_predictions.txt'

        # Write prediction summary and metrics to text file
        with output_file_path.open('w', encoding='utf-8') as file:
            # Prediction method and unique character count
            file.write(f'Prediction Method: {prediction_method_name}\n')
            file.write(f'Unique Character Count: {self.unique_character_count}\n\n')

            # Accuracy and validity metrics
            accuracy = evaluation_metrics['accuracy']
            validity = evaluation_metrics['validity']
            file.write(f'TOP1 ACCURACY: {accuracy[1]:.2%}\n')
            file.write(f'TOP2 ACCURACY: {accuracy[2]:.2%}\n')
            file.write(f'TOP3 ACCURACY: {accuracy[3]:.2%}\n')
            file.write(f'TOP1 VALIDITY: {validity[1]:.2%}\n')
            file.write(f'TOP2 VALIDITY: {validity[2]:.2%}\n')
            file.write(f'TOP3 VALIDITY: {validity[3]:.2%}\n\n')

            # Configuration details
            file.write(f'Train Size: {len(self.train_set)}, Test Size: {len(self.test_set)}\n')
            file.write(f'Vowel Ratio: {self.config.vowel_replacement_ratio}, '
                    f'Consonant Ratio: {self.config.consonant_replacement_ratio}\n\n')

            # Detailed prediction results
            for mod_word, miss_letter, orig_word, top_preds, cor_letter_rank in predictions:
                file.write(f'Test Word: {mod_word}, Correct Letter: {miss_letter}\n')
                file.write(f'Correct Letter Rank: {cor_letter_rank}\n')

                for rank, (pred_letter, confidence) in enumerate(top_preds, start=1):
                    reconstructed_word = mod_word.replace('_', pred_letter)
                    is_valid_word = reconstructed_word in self.all_words

                    file.write(f"Rank {rank}: '{pred_letter}' (Confidence: {confidence:.8f}), "
                            f"Valid: {is_valid_word}\n")
                
                file.write('\n')