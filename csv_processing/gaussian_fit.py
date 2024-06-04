import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import re

def read_ngram_data(directory):
    """
    Read n-gram data from .arpa files in the specified directory.

    Args:
        directory (Path): Directory containing .arpa files.

    Returns:
        dict: Dictionary with n-gram lengths as keys and their counts as values.
    """
    ngram_pattern = re.compile(r'ngram (\d+)=(\d+)')
    ngram_data = {}

    for file in directory.glob("*.arpa"):
        with file.open('r') as f:
            for line in f:
                match = ngram_pattern.search(line)
                if match:
                    ngram_length = int(match.group(1))
                    count = int(match.group(2))
                    if ngram_length in ngram_data:
                        ngram_data[ngram_length] += count
                    else:
                        ngram_data[ngram_length] = count
    return ngram_data

def gaussian_function(n, amplitude, mean, stddev):
    """
    Gaussian function used for curve fitting.

    Args:
        n (array-like): Independent variable (n-gram lengths).
        amplitude (float): Amplitude of the Gaussian function.
        mean (float): Mean of the Gaussian function.
        stddev (float): Standard deviation of the Gaussian function.

    Returns:
        array-like: Gaussian function values.
    """
    return amplitude * np.exp(-((n - mean)**2) / (2 * stddev**2))

def plot_ngram_distribution(ngram_data, dir_name, output_directory):
    """
    Plot and save the n-gram distribution with a Gaussian fit.

    Args:
        ngram_data (dict): Dictionary of n-gram lengths and counts.
        dir_name (str): Name of the directory (corpus) being processed.
        output_directory (Path): Directory to save the output plots.
    """
    n_values = np.array(sorted(ngram_data.keys()))
    ngram_counts = np.array([ngram_data[n] for n in n_values])

    # Provide initial guesses for the parameters
    initial_guesses = [max(ngram_counts), np.mean(n_values), np.std(n_values)]

    # Fit the Gaussian function to the data
    popt, pcov = curve_fit(gaussian_function, n_values, ngram_counts, p0=initial_guesses, maxfev=10000)

    # Generate data points for a smooth curve
    n_fit = np.linspace(min(n_values), max(n_values), 500)
    fit_curve = gaussian_function(n_fit, *popt)

    # Calculate the confidence intervals
    perr = np.sqrt(np.diag(pcov))
    lower_bound = gaussian_function(n_fit, *(popt - perr))
    upper_bound = gaussian_function(n_fit, *(popt + perr))

    # Plot the data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(n_values, ngram_counts, label='Data', color='blue', s=50)
    plt.plot(n_fit, fit_curve, label='Gaussian Fit', color='red', linewidth=2)
    plt.fill_between(n_fit, lower_bound, upper_bound, color='red', alpha=0.2)
    plt.xlabel('n-gram length', fontsize=14)
    plt.ylabel('Number of unique n-grams', fontsize=14)
    plt.title(f'Gaussian Fit to N-Gram Distribution ({dir_name})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Ensure the output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_directory / f'{dir_name}_ngram_gaussian_fit.png')
    plt.close()

    # Print the parameters of the Gaussian fit with their uncertainties
    print(f"Fitted parameters for {dir_name}: A = {popt[0]} ± {perr[0]}, mu = {popt[1]} ± {perr[1]}, sigma = {popt[2]} ± {perr[2]}")

def main():
    """
    Main function to process n-gram data from multiple directories,
    fit Gaussian functions, and plot the results.
    """
    base_directory = Path('data/models')
    output_directory = Path('data/outputs/gaussian_fit')
    dirs_to_process = [
        'brown',
        'CLMET3',
        'cmudict',
        'sorted_tokens_lampeter',
        'sorted_tokens_openEdges'
    ]

    combined_ngram_data = {}

    for dir_name in dirs_to_process:
        directory = base_directory / dir_name
        if directory.exists():
            ngram_data = read_ngram_data(directory)
            if ngram_data:
                # Accumulate data for combined fit
                for key, value in ngram_data.items():
                    if key in combined_ngram_data:
                        combined_ngram_data[key] += value
                    else:
                        combined_ngram_data[key] = value

                plot_ngram_distribution(ngram_data, dir_name, output_directory)
            else:
                print(f"No n-gram data found in {directory}")
        else:
            print(f"Directory {directory} does not exist")

    # Plot and save the combined Gaussian fit
    plot_ngram_distribution(combined_ngram_data, "combined_fit", output_directory)

if __name__ == "__main__":
    main()
