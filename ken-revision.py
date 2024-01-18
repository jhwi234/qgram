import logging
from pathlib import Path

from evaluation_class import EvaluateModel
from corpus_class import CorpusManager
from predictions_class import Predictions

# Configuration class for language model testing parameters. Change the testing inputs here.class Config:
class Config:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir else __file__).parent
        self.data_dir = self.base_dir / 'data'
        self.model_dir = self.data_dir / 'models'
        self.log_dir = self.data_dir / 'logs'
        self.corpus_dir = self.data_dir / 'corpora'
        self.output_dir = self.data_dir / 'outputs'
        self.text_dir = self.output_dir / 'texts'
        self.csv_dir = self.output_dir / 'csv'
        self.sets_dir = self.output_dir / 'sets'
        
        # Default values for other configurations
        self.seed = 42
        self.q_range = [6, 6]
        self.split_config = 0.5
        self.vowel_replacement_ratio = 0.2
        self.consonant_replacement_ratio = 0.8
        self.min_word_length = 3
        self.prediction_method_name = 'context_sensitive'
        self.log_level = logging.INFO

    # Logging Configuration: Setup log file and console output formats
    def setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logfile = self.log_dir / 'logfile.log'
        file_handler = logging.FileHandler(logfile, mode='a')
        file_format = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_format)

        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)

        logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    def create_directories(self):
        for directory in [self.data_dir, self.model_dir, self.log_dir, self.corpus_dir, self.output_dir, self.sets_dir, self.text_dir, self.csv_dir]:
            directory.mkdir(exist_ok=True)

def run(corpus_name, config):
    # Correctly use the static method from CorpusManager
    formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
    logging.info(f'Processing {formatted_corpus_name} Corpus')
    logging.info('-' * 40)

    # Create an instance of CorpusManager
    corpus_manager = CorpusManager(formatted_corpus_name, config)
    CorpusManager.add_to_global_corpus(corpus_manager.corpus) 

    # Initialize EvaluateModel with the corpus manager
    eval_model = EvaluateModel(corpus_manager)

    # Retrieve the prediction method from the Predictions object
    prediction_method = getattr(eval_model.predictor, config.prediction_method_name)
    evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)
    logging.info(f'Evaluated with: {prediction_method.__name__}')

    # Log the accuracy and validity results
    accuracy = evaluation_metrics['accuracy']
    validity = evaluation_metrics['validity']
    logging.info(f'Model evaluation completed for: {corpus_name}')
    logging.info(f'TOP1 ACCURACY: {accuracy[1]:.2%} | TOP1 VALIDITY: {validity[1]:.2%}')
    logging.info(f'TOP2 ACCURACY: {accuracy[2]:.2%} | TOP2 VALIDITY: {validity[2]:.2%}')
    logging.info(f'TOP3 ACCURACY: {accuracy[3]:.2%} | TOP3 VALIDITY: {validity[3]:.2%}')

    # Save the predictions to CSV and text files
    eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
    eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)
    eval_model.save_recall_precision_stats()

    logging.info('-' * 40)

def main():
    config = Config()
    config.setup_logging()
    config.create_directories()

    corpora = ['cmudict', 'brown', 'CLMET3.txt', 'reuters', 'gutenberg', 'inaugural']
    for corpus_name in corpora:
        run(corpus_name, config)

    # Create and evaluate the mega-corpus
    mega_corpus_name = 'mega_corpus'
    with open(config.corpus_dir / f'{mega_corpus_name}.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(CorpusManager.unique_words_all_corpora))

    run(mega_corpus_name, config)

if __name__ == '__main__':
    main()
