import logging
import regex
import unicodedata
from bs4 import BeautifulSoup

class TextProcessorError(Exception):
    """Custom exception class for TextProcessor errors."""
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)

logging.basicConfig(level=logging.ERROR)

class TextProcessor:
    """
    A class used to process text for Middle and Old English corpora. It includes methods for removing HTML tags,
    non-linguistic strings, and tokenizing the text.

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
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the TextProcessor object.
        """
        self.tokenizer_pattern = (r'\b[\Þ\þ\Ƿ\ƿ\Ð\ð\Ȝ\ȝ\Æ\æ\Œ\œ\⁊a-zA-Z][\Þ\þ\Ƿ\ƿ\Ð\ð\Ȝ\ȝ\Æ\æ\Œ\œ\⁊\'a-zA-Z-]*\b')
        self.non_linguistic_patterns = [
            regex.compile(r'\[[^\]]*\]'),          # Matches anything in square brackets
            regex.compile(r'\{[^\}]*\}'),          # Matches anything in curly brackets
            regex.compile(r'\[[sS][iI][cC]\]'),    # Matches [sic] in any case
            regex.compile(r'l\d+'),                # Matches 'l' followed by any digits
            regex.compile(r'O\d+'),                # Matches 'O' followed by any digits
            regex.compile(r'\d{2,}-\d{2,}'),       # Matches number ranges like dates
        ]

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
            The text without HTML tags.
        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_non_linguistic_strings(self, text):
        """
        Removes common non-linguistic strings from the text using regular expressions.

        Parameters
        ----------
        text : str
            Text from which to remove non-linguistic strings.
        
        Returns
        -------
        str
            The text with non-linguistic strings removed.
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
            Line of text to check.
        
        Returns
        -------
        bool
            True if the line is valid, False otherwise.
        """
        return not (line.startswith('#') or line.isupper())

    def tokenize_text(self, text):
        """
        Tokenizes the given text by identifying words, contractions, hyphenated words, 
        and words with Old and Middle English characters.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        list
            A list of tokens.
        """
        # Ensure input is a string
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        # Normalize text to NFKC (Compatibility Composition)
        normalized_text = unicodedata.normalize('NFKC', text)

        # Initialize an empty list to store tokens
        tokens = []

        for line in normalized_text.split('\n'):
            if self.is_valid_line(line):
                line = line.lower()
                matches_found = False  # To indicate if any matches were found
                for token in regex.finditer(self.tokenizer_pattern, line, overlapped=True):
                    matches_found = True
                    tokens.append(token.group())

        return tokens

    def process_text(self, text):
        """
        Processes the given text by removing HTML tags, non-linguistic strings, and tokenizing the text.

        Parameters
        ----------
        text : str
            Text to process.

        Returns
        -------
        list
            A list of tokens from the processed text.
        """
        text_without_html = self.remove_html_tags(text)
        cleaned_text = self.remove_non_linguistic_strings(text_without_html)
        return self.tokenize_text(cleaned_text)

    def process_file(self, filepath):
        """
        Processes the content of a given file by applying the text processing methods.

        Parameters
        ----------
        filepath : str
            The path to the file to process.

        Returns
        -------
        list
            A list of tokens from the processed file.
        """
        if not isinstance(filepath, str):
            raise TextProcessorError("File path must be a string")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()
            return self.process_text(file_content)
        except FileNotFoundError as fnf_error:
            logging.error(f"The file {filepath} was not found: {fnf_error}")
            raise TextProcessorError(f"The file {filepath} was not found") from fnf_error
        except IOError as io_error:
            logging.error(f"An IO error occurred: {io_error}")
            raise TextProcessorError("An IO error occurred") from io_error
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise TextProcessorError("An unexpected error occurred") from e