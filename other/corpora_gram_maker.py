from pathlib import Path
import csv
from collections import Counter

def extract_qgrams(word, max_q):
    """Generates up to max_q q-grams for the given word."""
    return [word[i:i+q] for q in range(1, max_q + 1) for i in range(len(word) - q + 1)]

def generate_and_save_qgrams_frequency_table(filename, max_q):
    """Reads a CSV, generates a frequency table of q-grams, and saves it to a file in a specified directory."""
    csv_dir = Path('data/outputs/csv/')
    csv_file_path = csv_dir / filename

    # Ensure the CSV file exists before proceeding
    if not csv_file_path.exists():
        return "CSV file does not exist."
    
    # Prepare the qgrams directory
    qgrams_dir = Path('data/outputs/qgrams')
    qgrams_dir.mkdir(parents=True, exist_ok=True)

    # Compute the frequency table directly from the CSV
    with csv_file_path.open(newline='', encoding='utf-8') as csvfile:
        frequency_table = Counter(
            qgram
            for row in csv.DictReader(csvfile)
            for qgram in extract_qgrams(row['Original_Word'], max_q)
        )

    # Sort the frequency table by frequency in descending order
    sorted_frequency_table = sorted(frequency_table.items(), key=lambda item: item[1], reverse=True)

    # Path for the output text file
    output_file_path = qgrams_dir / f"sorted_{max_q}grams_frequency_table.txt"
    
    # Save the sorted frequency table to a text file
    with output_file_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(f"{qgram}: {frequency}" for qgram, frequency in sorted_frequency_table))

    return f"Sorted {max_q}-grams frequency table saved successfully to {output_file_path}"

if __name__ == "__main__":
    csv_filename = "brown_context_sensitive_split0.5_qrange6-6_prediction.csv"
    result = generate_and_save_qgrams_frequency_table(csv_filename, 6)
    print(result)
