import zipfile
from io import BytesIO
import os
from TextProcessor import TextProcessor  # Import the class from your TextProcessor.py file

# Initialize an instance of the TextProcessor
text_processor = TextProcessor()

def extract_text_from_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path) as zip_file:
        for file_info in zip_file.infolist():
            if file_info.filename.endswith('.txt'):
                with zip_file.open(file_info) as file:
                    # Decoding the file contents and handling XML/HTML-like entities
                    yield file_info.filename, text_processor.decode_entities(file.read().decode('utf-8'))

def process_and_collect_tokens(zip_path, inner_zip_path):
    unique_tokens = set()

    with zipfile.ZipFile(zip_path) as outer_zip:
        for inner_zip_info in outer_zip.infolist():
            if inner_zip_info.filename.startswith(inner_zip_path) and inner_zip_info.filename.endswith('.zip'):
                with outer_zip.open(inner_zip_info) as inner_zip_file:
                    inner_zip_bytes = BytesIO(inner_zip_file.read())
                    with zipfile.ZipFile(inner_zip_bytes) as inner_zip:
                        for txt_filename, text in extract_text_from_zip(inner_zip):
                            # Process the text to remove tags, page indicators, and get tokens
                            text = text_processor.remove_tags_and_metadata(text)
                            file_tokens = text_processor.tokenize(text)
                            unique_tokens.update(file_tokens)

    return unique_tokens

def save_tokens(tokens, output_path):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for token in sorted(tokens):
            output_file.write(f"{token}\n")

if __name__ == "__main__":
    root_dir = 'The Corpus of Late Modern English Texts'
    zip_path = os.path.join(root_dir, 'clmet3_1.zip')
    inner_zip_path = 'clmet/corpus/txt/plain/'
    
    unique_tokens = process_and_collect_tokens(zip_path, inner_zip_path)
    
    output_path = os.path.join(root_dir, 'unique_tokens.txt')
    save_tokens(unique_tokens, output_path)
    
    print(f"Extraction and tokenization complete. {len(unique_tokens)} unique tokens written to {output_path}")