from pathlib import Path
from TextProcessor import TextProcessor

def process_corpora(corpora_base_path, corpora_name):
    # Determine if the corpora require recursive directory traversal
    corpora_path = corpora_base_path / corpora_name
    if corpora_name in ['clmet', 'lampeter', 'openEdges']:
        text_files = list(corpora_path.rglob('*.txt'))  # Recursive search for nested directories
    else:
        text_files = list(corpora_path.glob('*.txt'))  # Direct search for current directory files
    
    print(f"Found {len(text_files)} files to process in {corpora_name}.")

    text_processor = TextProcessor()
    all_tokens = set()
    all_tokens_list = []  # List to store all tokens including duplicates

    for i, file_path in enumerate(text_files, 1):
        print(f"Processing file {i}/{len(text_files)}: {file_path.name} in {corpora_name}")
        try:
            tokens = text_processor.process_file(str(file_path))
            all_tokens.update(tokens)  # Update the set with new tokens
            all_tokens_list.extend(tokens)  # Extend the list with new tokens
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # Sort and deduplicate tokens
    sorted_tokens = sorted(all_tokens)

    # Write the sorted, unique tokens to output file specific to the corpora
    sorted_output_path = f'sorted_tokens_{corpora_name}.txt'
    try:
        with open(sorted_output_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(sorted_tokens))
        print(f"Total unique tokens written to {sorted_output_path}: {len(sorted_tokens)}")
    except Exception as e:
        print(f"An error occurred while writing to {sorted_output_path}: {e}")

    # Write all tokens to another file specific to the corpora
    all_tokens_output_path = f'all_tokens_{corpora_name}.txt'
    try:
        with open(all_tokens_output_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(all_tokens_list))
        print(f"Total tokens written to {all_tokens_output_path}: {len(all_tokens_list)}")
    except Exception as e:
        print(f"An error occurred while writing to {all_tokens_output_path}: {e}")

def main():
    base_path = Path("Historical English corpora").resolve()
    for corpora_name in ['clmet', 'lampeter', 'openEdges']:
        process_corpora(base_path, corpora_name)

if __name__ == '__main__':
    main()
