from pathlib import Path
from TextProcessor import TextProcessor

def main():
    text_processor = TextProcessor()
    directory_path = Path("Historical English corpora/clmet/txt").resolve()
    text_files = list(directory_path.glob('*.txt'))
    print(f"Found {len(text_files)} files to process.")

    all_tokens = set()
    all_tokens_list = []  # List to store all tokens including duplicates

    for i, file_path in enumerate(text_files, 1):
        print(f"Processing file {i}/{len(text_files)}: {file_path.name}")
        try:
            tokens = text_processor.process_file(str(file_path))
            all_tokens.update(tokens)  # Update the set with new tokens
            all_tokens_list.extend(tokens)  # Extend the list with new tokens
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    print("Sorting and deduplicating tokens...")
    sorted_tokens = sorted(all_tokens)

    # Write the sorted, unique tokens to output file
    sorted_output_path = 'sorted_tokens.txt'
    try:
        with open(sorted_output_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(sorted_tokens))
        print(f"Total unique tokens written to {sorted_output_path}: {len(sorted_tokens)}")
    except Exception as e:
        print(f"An error occurred while writing to {sorted_output_path}: {e}")

    # Write all tokens to another file
    all_tokens_output_path = 'all_tokens.txt'
    try:
        with open(all_tokens_output_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(all_tokens_list))
        print(f"Total tokens written to {all_tokens_output_path}: {len(all_tokens_list)}")
    except Exception as e:
        print(f"An error occurred while writing to {all_tokens_output_path}: {e}")

    print("Processing complete. Check the output files for results.")

if __name__ == '__main__':
    main()
