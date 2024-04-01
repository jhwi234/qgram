from pathlib import Path
import csv
from collections import Counter

def extract_qgrams(word, max_q):
    """Generate q-grams of lengths 1 through max_q for a given word."""
    return [word[i:i+q] for q in range(1, max_q + 1) for i in range(len(word) - q + 1)]

def read_csv_and_generate_qgrams(csv_file_path, max_q):
    """Read a CSV file and generate q-grams for each 'Original_Word'."""
    with csv_file_path.open(newline='', encoding='utf-8') as csvfile:
        return [
            qgram
            for row in csv.DictReader(csvfile)
            for qgram in extract_qgrams(row['Original_Word'], max_q)
        ]

def save_frequency_table(sorted_frequency_table, output_file_path):
    """Save the sorted frequency table to a file."""
    content = "\n".join(f"{qgram}: {frequency}" for qgram, frequency in sorted_frequency_table)
    output_file_path.write_text(content, encoding="utf-8")

def generate_and_save_qgrams_frequency_table(filename, max_q):
    """Process a CSV file to generate and save a q-grams frequency table."""
    csv_file_path = Path('data/outputs/csv') / filename

    if not csv_file_path.is_file():
        return "CSV file does not exist."
    
    # Extract the first part of the filename to use in the output filename
    filename_without_ext = csv_file_path.stem

    qgrams_dir = Path('data/outputs/qgrams')
    qgrams_dir.mkdir(parents=True, exist_ok=True)

    # Generate q-grams and create a frequency table
    qgrams = read_csv_and_generate_qgrams(csv_file_path, max_q)
    frequency_table = Counter(qgrams)
    sorted_table = sorted(frequency_table.items(), key=lambda item: item[1], reverse=True)

    output_file_path = qgrams_dir / f"{filename_without_ext}_{max_q}grams_frequency_table.txt"
    save_frequency_table(sorted_table, output_file_path)

    return f"Sorted {max_q}-grams frequency table saved successfully to {output_file_path}"

def main():
    csv_filename = "brown_context_sensitive_split0.5_qrange6-6_prediction.csv"
    result = generate_and_save_qgrams_frequency_table(csv_filename, 6)
    print(result)

if __name__ == "__main__":
    main()
