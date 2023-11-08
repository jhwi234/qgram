"""
A module for processing text for Middle and Old English corpora. It removes HTML tags and non-linguistic strings,
tokenizes the text, and then processes all text files in a given directory and its subdirectories.
"""

import os
import glob
import re
from bs4 import BeautifulSoup
import regex
import unicodedata
import logging
import csv
from collections import Counter
import sqlite3

class TextProcessor:
    """
    A class used to process text for Middle and Old English corpora, interact with a database to store unique tokens,
    and write processed tokens to files.

    Attributes
    ----------
    root_dir : str
        The root directory where the text files to be processed are located.

    Methods
    -------
    remove_html_tags(text: str) -> str
        Removes HTML tags from the given text using BeautifulSoup.

    remove_non_linguistic_strings(text: str) -> str
        Removes common non-linguistic strings from the text using regular expressions.

    is_valid_line(line: str) -> bool
        Checks if a line of text is valid (i.e., not metadata or a title), by ensuring it does not start with a hash or is entirely in uppercase.

    tokenize_text(text: str) -> set
        Tokenizes the given text by identifying words, contractions, hyphenated words, and words with Old and Middle English characters, and returns a set of unique tokens.

    process_text(text: str) -> set
        Processes the given text by removing HTML tags, non-linguistic strings, and tokenizing the text. Returns a set of unique tokens.

    process_files()
        Processes all text files in the given root directory and its subdirectories by reading them, cleaning them, and tokenizing the contents. The unique tokens are stored in a database and batch files.

    initialize_database()
        Sets up a SQLite database connection and creates a table for tokens if it does not exist.

    is_unique_token(token: str) -> bool
        Queries the database to check if a token is unique, i.e., not already present in the database.

    add_token_to_database(token: str)
        Adds a token to the database if it's not already present.

    process_batch(batch: list)
        Processes a batch of tokens by adding unique ones to the database and writes them to a CSV file. Each batch is stored in a separate file within a 'batches' subdirectory.

    __del__()
        Closes the database connection when the TextProcessor object is deleted.
    """
    def __init__(self, root_dir):
        """
        Constructs all the necessary attributes for the TextProcessor object.

        Parameters
        ----------
            root_dir : str
                root directory where the text files to be processed are located
        """
        self.root_dir = root_dir
        self.tokenizer_pattern = r'(?:\b|(?<=-))[\p{L}&&[^\p{N}]]+(?=\b|(?=-))'
        self.non_linguistic_patterns = [
            regex.compile(r'\[[^\]]*\]'),          # Matches anything in square brackets
            regex.compile(r'\{[^\}]*\}'),          # Matches anything in curly brackets
            regex.compile(r'\[[sS][iI][cC]\]'),    # Matches [sic] in any case
            regex.compile(r'l\d+'),                # Matches 'l' followed by any digits
            regex.compile(r'O\d+'),                # Matches 'O' followed by any digits
            regex.compile(r'Page \d+'),            # Matches 'Page' followed by any digits
            regex.compile(r'ß \d+'),               # Matches 'ß' followed by any digits
            regex.compile(r'Lin\.\s+\d+'),         # Matches 'Lin.' followed by any digits
            regex.compile(r'[Pp]ag\.\s+\d+'),      # Matches 'Pag.' or 'pag.' followed by any digits
            regex.compile(r'\d{2,}-\d{2,}'),       # Matches number ranges like dates
        ]
        self.batch_count = 0
        self.batch_output_dir = os.path.join(self.root_dir, 'batches')
        os.makedirs(self.batch_output_dir, exist_ok=True)
        self.db_path = os.path.join(self.root_dir, 'tokens.db')
        self.initialize_database()

    def remove_html_tags(self, text):
        """
        Removes HTML tags from the given text using BeautifulSoup.

        Parameters
        ----------
            text : str
                Text from which to remove HTML tags.
        
        Returns
        -------
            str
                The cleaned text.
        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_non_linguistic_strings(self, text):
        """
        Removes common non-linguistic strings from the text.

        Parameters
        ----------
            text : str
                text from which to remove non-linguistic strings
        
        Returns
        -------
            str
                text with non-linguistic strings removed
        """
        for pattern in self.non_linguistic_patterns:
            text = pattern.sub('', text)
        return text.strip()

    def is_valid_line(self, line):
        """
        Checks if a line of text is valid (i.e., not metadata or a title).

        Parameters
        ----------
            line : str
                line of text to check
        
        Returns
        -------
            bool
                True if line of text is valid, False otherwise
        """
        return not (line.startswith('#') or line.isupper())

    def tokenize_text(self, text):
        """
        Tokenizes the given text by separating words, contractions, hyphenated words, and words with Old and Middle English characters.
        It normalizes the input to ensure consistent encoding and filters lines based on the is_valid_line method.

        Parameters
        ----------
            text : str
                Text to tokenize.

        Returns
        ------
            set
                A set of unique tokens.
        """
        # Ensure input is a string
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        # Normalize text to NFKC (Compatibility Composition)
        normalized_text = unicodedata.normalize('NFKC', text)

        # Initialize an empty set to store unique tokens
        unique_tokens = set()

        # Process each line individually
        for line in normalized_text.split('\n'):
            if self.is_valid_line(line):
                line = line.lower()
                for token in regex.finditer(self.tokenizer_pattern, line, overlapped=True):
                    unique_tokens.add(token.group())

        for token in unique_tokens:
            yield token

    def process_text(self, text):
        """
        Processes the given text by removing HTML tags, non-linguistic strings, and tokenizing the text.

        Parameters
        ----------
            text : str
                text to process

        Returns
        -------
            set
                set of unique tokens

        Raises
        ------
            ValueError
                If input text is not of string type
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        
        text_without_html = self.remove_html_tags(text)
        cleaned_text = self.remove_non_linguistic_strings(text_without_html)
        tokens = self.tokenize_text(cleaned_text)
        
        return tokens

    def process_files(self):
        """
        Processes all text files within the root directory and its subdirectories.
        """
        file_data = {}
        batch_size = 10
        batch = []

        for file_path in glob.glob(os.path.join(self.root_dir, '**/*.txt'), recursive=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        cleaned_line = self.remove_non_linguistic_strings(self.remove_html_tags(line))
                        for token in self.tokenize_text(cleaned_line):
                            batch.append(token)
                            if len(batch) >= batch_size:
                                self.process_batch(batch)
                                batch = []

                if batch:
                    self.process_batch(batch)

            except IOError as e:
                logging.error(f"An IOError occurred while processing the file {file_path}: {e.strerror}")

    def initialize_database(self):
        """
        Initializes the database for storing tokens.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS tokens (token TEXT PRIMARY KEY)')
        self.conn.commit()

    def is_unique_token(self, token):
        """
        Checks if a token is already in the database.

        Parameters
        ----------
            token : str
                The token to check for uniqueness.

        Returns
        -------
            bool
                True if token is unique, False otherwise.
        """
        self.cursor.execute('SELECT token FROM tokens WHERE token = ?', (token,))
        return self.cursor.fetchone() is None

    def add_token_to_database(self, token):
        """
        Adds a token to the database if it's not already present.

        Parameters
        ----------
            token : str
                The token to add to the database.
        """
        if self.is_unique_token(token):
            self.cursor.execute('INSERT INTO tokens (token) VALUES (?)', (token,))
            self.conn.commit()

    def process_batch(self, batch):
        """
        Processes a batch of tokens by adding them to the database and writing them to a CSV file if they are unique.
        
        Parameters
        ----------
            batch : list
                Batch of tokens to process.
        """
        for token in batch:
            self.add_token_to_database(token)

        # Define a file name pattern for batch files
        batch_file_name = f"batch_{self.batch_count:04d}.csv"
        batch_file_path = os.path.join(self.batch_output_dir, batch_file_name)

        # Write the batch of tokens to a CSV file
        with open(batch_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['token'])
            for token in batch:
                if self.is_unique_token(token):
                    writer.writerow([token])

        self.batch_count += 1

        return batch_file_path

    def __del__(self):
        """
        Destructor to close the database connection when the object is deleted.
        """
        self.conn.close()