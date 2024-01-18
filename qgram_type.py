import logging
from pathlib import Path

from evaluation_class import EvaluateModel
from corpus_class import CorpusManager
from predictions_class import Predictions

class Config:
    # Configuration class for setting up directories and default parameters
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir else __file__).parent
        self._set_directories()
        self._set_values()

    def _set_directories(self):
        # Setup various directories needed for the application
        self.data_dir = self.base_dir / 'data'
        self.model_dir = self.data_dir / 'models'
        self.log_dir = self.data_dir / 'logs'
        self.corpus_dir = self.data_dir / 'corpora'
        self.output_dir = self.data_dir / 'outputs'
        self.text_dir = self.output_dir / 'texts'
        self.csv_dir = self.output_dir / 'csv'
        self.sets_dir = self.output_dir / 'sets'

    def _set_values(self):
        # Values for testing
        self.seed = 42
        self.q_range = [6, 6]
        self.split_config = 0.5
        self.vowel_replacement_ratio = 0.2
        self.consonant_replacement_ratio = 0.8
        self.min_word_length = 3
        self.prediction_method_name = 'context_sensitive'
        self.log_level = logging.INFO

    def setup_logging(self):
        # Setup logging with file and console handlers
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logfile = self.log_dir / 'logfile.log'
        file_handler = logging.FileHandler(logfile, mode='a')
        file_format = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_format)

        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)

        logging.basicConfig(level=self.log_level, handlers=[file_handler, console_handler])

    def create_directories(self):
        # Create necessary directories if they don't exist
        for directory in [self.data_dir, self.model_dir, self.log_dir, self.corpus_dir, self.output_dir, self.sets_dir, self.text_dir, self.csv_dir]:
            directory.mkdir(exist_ok=True)

def run(corpus_name, config):
    # Use the static method from CorpusManager to format the corpus name
    formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
    logging.info(f'Processing {formatted_corpus_name} Corpus')
    logging.info('-' * 40)

    # Initialize CorpusManager with the formatted corpus name and configuration settings
    corpus_manager = CorpusManager(formatted_corpus_name, config)

    # Add unique words from the current corpus to the global corpus
    CorpusManager.add_to_global_corpus(corpus_manager.corpus)

    # Create an EvaluateModel instance, passing the CorpusManager instance
    eval_model = EvaluateModel(corpus_manager)

    # Retrieve the prediction method based on the configuration
    prediction_method = getattr(eval_model.predictor, config.prediction_method_name)

    # Evaluate character predictions using the selected prediction method
    evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)

    # Log the results of the evaluation for accuracy and validity
    logging.info(f'Evaluated with: {prediction_method.__name__}')
    logging.info(f'Model evaluation completed for: {corpus_name}')
    for i in range(1, 4):
        logging.info(f'TOP{i} ACCURACY: {evaluation_metrics["accuracy"][i]:.2%} | TOP{i} VALIDITY: {evaluation_metrics["validity"][i]:.2%}')

    # Export the prediction details and summary statistics to CSV and text files
    eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
    eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)

    # Save recall and precision statistics
    eval_model.save_recall_precision_stats(evaluation_metrics)

    logging.info('-' * 40)

def main():
    # Setup logging and create necessary directories
    config = Config()
    config.setup_logging()
    config.create_directories()

    # Iterating over each corpus for processing
    corpora = ['cmudict', 'brown', 'CLMET3.txt', 'reuters', 'gutenberg', 'inaugural']
    for corpus_name in corpora:
        run(corpus_name, config)

    # Create, process, and evaluate the mega-corpus
    mega_corpus_name = 'mega_corpus'
    with open(config.corpus_dir / f'{mega_corpus_name}.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(CorpusManager.unique_words_all_corpora))

    run(mega_corpus_name, config)

if __name__ == '__main__':
    main()
