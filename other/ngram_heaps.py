import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

def read_ngram_frequencies(file_path):
    """Read the n-gram frequencies from the file."""
    frequencies = []
    if not file_path.exists():
        print(f"Error: The file {file_path} does not exist.")
        return frequencies
    
    try:
        with file_path.open('r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2 and parts[1].isdigit():
                    frequencies.append(int(parts[1]))
                else:
                    print(f"Skipping invalid line: {line.strip()}")
        print(f"Read {len(frequencies)} n-gram frequencies from {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return frequencies

def plot_heap_law(frequencies):
    """Plot the number of distinct n-grams (types) against the total number of n-grams (tokens) and fit a power law."""
    if not frequencies:
        print("No frequencies to plot.")
        return

    frequencies = np.array(frequencies)
    total_tokens = np.cumsum(frequencies)
    distinct_types = np.arange(1, len(frequencies) + 1)

    # Fit a power law to the data
    def heap_law(N, k, beta):
        return k * N ** beta

    # Initial guess for parameters k and beta
    initial_guess = [1.0, 0.5]
    try:
        popt, pcov = curve_fit(heap_law, total_tokens, distinct_types, p0=initial_guess, maxfev=10000)
        print(f"Fitted parameters: k={popt[0]:.2f}, β={popt[1]:.2f}")
    except RuntimeError as e:
        print(f"An error occurred during curve fitting: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(total_tokens, distinct_types, marker='o', linestyle='none', label='Data')
    plt.plot(total_tokens, heap_law(total_tokens, *popt), label=f'Fit: k={popt[0]:.2f}, β={popt[1]:.2f}', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Total Number of Tokens (n-grams)')
    plt.ylabel('Number of Distinct Types (n-grams)')
    plt.title('Heap\'s Law Analysis of Character n-grams')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file_path = Path("character_ngrams_frequency.txt")

    frequencies = read_ngram_frequencies(file_path)
    
    if frequencies:
        plot_heap_law(frequencies)

if __name__ == "__main__":
    main()
