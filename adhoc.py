def find_distinct_letters(file_path: str):
    distinct_letters = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Extract words from each line
            words = line.split()
            for word in words:
                # Add each letter to the set
                distinct_letters.update(letter.lower() for letter in word if letter.isalpha())

    # Print out all the distinct letters
    print('Distinct letters:', ''.join(sorted(distinct_letters)))

# Call the function with your file path
find_distinct_letters('CLMET3_words.txt')