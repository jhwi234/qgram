# Import required modules
import os

# Define the function to count strings in files starting with "all_" in a specific directory
def count_strings_in_files(directory_path='data/corpora/'):
    # Change the current working directory to the specified path
    os.chdir(directory_path)
    
    # List all files in the directory
    all_files = os.listdir()
    
    # Filter files that start with "all_" and end with ".txt"
    target_files = [file for file in all_files if file.startswith('sorted_') and file.endswith('.txt')]
    
    # Dictionary to hold the count of strings for each file
    strings_count = {}
    
    # Loop through each target file
    for file in target_files:
        # Initialize a count variable for the current file
        count = 0
        
        # Open the file and read its contents
        with open(file, 'r') as f:
            # Read the file line by line
            for line in f:
                # Split the line into strings assuming they are separated by spaces
                strings = line.split()
                # Update the count with the number of strings in the current line
                count += len(strings)
                
        # Update the dictionary with the count for the current file
        strings_count[file] = count
    
    # Return the dictionary containing the count of strings for each file
    return strings_count

# Note: The function call is commented out to prevent execution in this environment
print(count_strings_in_files())
