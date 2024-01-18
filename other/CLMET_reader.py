# import necessary modules
import os
import glob
from TextProcessor import TextProcessor

def main():
    # Initialize the TextProcessor
    text_processor = TextProcessor()

    # Define the directory path as absolute to avoid issues when running from different directories
    directory_path = os.path.abspath("Historical English corpora/clmet/txt")

    # Use glob to find all .txt files in the directory
    text_files = glob.glob(os.path.join(directory_path, '*.txt'))
    total_files = len(text_files)
    print(f"Found {total_files} files to process.")

    # Check if temp_tokens.txt already exists and remove it
    temp_tokens_path = 'temp_tokens.txt'
    if os.path.exists(temp_tokens_path):
        os.remove(temp_tokens_path)
    
    # Process each file one by one
    for i, file_path in enumerate(text_files, 1):
        print(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
        try:
            # Process the content of the file
            tokens = text_processor.process_file(file_path)
            print(f"Found {len(tokens)} tokens.")  # Debugging line

            # Write the tokens to a temporary file, appending them
            with open(temp_tokens_path, 'a', encoding='utf-8') as temp_file:
                for token in tokens:
                    temp_file.write(f"{token}\n")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # After processing all files, we'll now sort and deduplicate the tokens
    try:
        print("Sorting and deduplicating tokens...")
        with open(temp_tokens_path, 'r', encoding='utf-8') as temp_file:
            all_tokens = set(line.strip() for line in temp_file)  # Use strip() here
        print(f"Number of unique tokens before sorting: {len(all_tokens)}")  # Debugging line
    except Exception as e:
        print(f"An error occurred while reading from the temporary file: {e}")
        return

    # Sort the set of all tokens (which also deduplicates them)
    try:
        sorted_tokens = sorted(all_tokens)
    except Exception as e:
        print(f"An error occurred during sorting: {e}")
        return

    # Write the sorted, unique tokens to the final output file
    output_path = 'sorted_tokens.txt'
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(sorted_tokens))
        print(f"Total unique tokens written: {len(sorted_tokens)}")  # Debugging line
    except Exception as e:
        print(f"An error occurred while writing to the final output file {output_path}: {e}")
        return
    finally:
        # Clean up the temporary file if it exists
        if os.path.exists(temp_tokens_path):
            os.remove(temp_tokens_path)

    print(f"Processing complete. The sorted, unique tokens have been saved to {output_path}")

if __name__ == "__main__":
    main()