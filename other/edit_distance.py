import numpy as np

class EditDistance:
    def levenshtein_distance(self, s1, s2):
        # Check for equality and handle the trivial case where both strings are identical.
        if s1 == s2: return 0

        # Ensure s1 is the shorter string to optimize memory usage in the dynamic programming table.
        if len(s1) > len(s2): s1, s2 = s2, s1

        # Initialize the previous row of the dynamic programming table. This represents the number of edits
        # needed to transform an empty string into the first i characters of s1.
        prev_row = np.arange(len(s1) + 1)

        # Iterate over each character in the second string.
        for c2 in s2:
            # Store the top-left value (from the previous iteration) and increment the first cell
            # which represents transforming s1 into the first character of the current substring of s2.
            old_value, prev_row[0] = prev_row[0], prev_row[0] + 1

            # Iterate over each character in the first string.
            for i1, c1 in enumerate(s1):
                # Calculate the cost for substitution. It's 0 if characters are the same, else 1.
                # Compare this with the costs for insertion and deletion, and pick the minimum.
                # old_value represents the substitution cost (top-left cell).
                # prev_row[i1] + 1 represents the deletion cost (top cell).
                # prev_row[i1 + 1] + 1 represents the insertion cost (left cell).
                new_value = min(old_value + (c1 != c2), prev_row[i1] + 1, prev_row[i1 + 1] + 1)
                
                # Update old_value for the next iteration and set the calculated minimum edit distance
                # for the current cell.
                old_value, prev_row[i1 + 1] = prev_row[i1 + 1], new_value

        # After completing the iterations, the last element of prev_row contains the Levenshtein distance.
        return prev_row[-1]

    def damerau_levenshtein_distance(self, s1, s2):
        """Damerau-Levenshtein distance calculation."""
        if s1 == s2: return 0
        if len(s1) > len(s2): s1, s2 = s2, s1

        prev_prev_row, prev_row = np.zeros(len(s2) + 1, dtype=int), np.arange(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = np.zeros(len(s2) + 1, dtype=int)
            current_row[0] = i + 1
            for j, c2 in enumerate(s2):
                insertions = current_row[j] + 1
                deletions = prev_row[j + 1] + 1
                substitutions = prev_row[j] + (c1 != c2)
                cost = min(insertions, deletions, substitutions)

                if i > 0 and j > 0 and c1 == s2[j - 1] and s1[i - 1] == c2:
                    transpositions = prev_prev_row[j - 1] + 1
                    cost = min(cost, transpositions)

                current_row[j + 1] = cost

            prev_prev_row, prev_row = prev_row, current_row

        return prev_row[-1]

    def hamming_distance(self, s1, s2):
        """Hamming distance calculation."""
        if len(s1) != len(s2): raise ValueError("Hamming distance requires equal length strings")
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def jaro_distance(self, s1, s2):
        """Jaro distance calculation."""
        if s1 == s2: return 1.0  # Quick return for identical strings
        if not s1 or not s2: return 0.0  # Handle empty strings

        len_s1, len_s2 = len(s1), len(s2)
        match_distance = (max(len_s1, len_s2) // 2) - 1

        s1_matches = np.zeros(len_s1, dtype=bool)
        s2_matches = np.zeros(len_s2, dtype=bool)
        matches = 0

        for i, c1 in enumerate(s1):
            start, end = max(0, i - match_distance), min(i + match_distance + 1, len_s2)
            for j, c2 in enumerate(s2[start:end], start):
                if not s1_matches[i] and not s2_matches[j] and c1 == c2:
                    s1_matches[i] = s2_matches[j] = True
                    matches += 1
                    break

        if not matches: return 0.0

        transpositions = sum(s1[i] != s2[j] for i, j in zip(np.where(s1_matches)[0], np.where(s2_matches)[0]))
        return ((matches / len_s1) + (matches / len_s2) + ((matches - transpositions / 2) / matches)) / 3.0

    def jaro_winkler_distance(self, s1, s2, p=0.1, max_prefix=4):
        """Jaro-Winkler distance calculation."""
        jaro_dist = self.jaro_distance(s1, s2)
        if jaro_dist < 0.7: return jaro_dist  # Threshold check
        prefix = sum(c1 == c2 for c1, c2 in list(zip(s1, s2))[:max_prefix])
        return jaro_dist + prefix * p * (1 - jaro_dist)

    def _initialize_dp_table(self, m, n):
        """Initialize the dynamic programming table for LCS computation."""
        return [[0] * (n + 1) for _ in range(m + 1)]

    def _calculate_lcs_length(self, X, Y, dp):
        """Calculate length of LCS using dynamic programming."""
        m, n = len(X), len(Y)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    def _find_all_lcs_sequences(self, X, Y, m, n, dp, memo):
        """Find all LCS sequences."""
        if m == 0 or n == 0:
            return set([""])
        if (m, n) in memo:
            return memo[(m, n)]

        sequences = set()
        if X[m - 1] == Y[n - 1]:
            lcs = self._find_all_lcs_sequences(X, Y, m - 1, n - 1, dp, memo)
            sequences = {Z + X[m - 1] for Z in lcs}
        else:
            if dp[m - 1][n] >= dp[m][n - 1]:
                sequences.update(self._find_all_lcs_sequences(X, Y, m - 1, n, dp, memo))
            if dp[m][n - 1] >= dp[m - 1][n]:
                sequences.update(self._find_all_lcs_sequences(X, Y, m, n - 1, dp, memo))

        memo[(m, n)] = sequences
        return sequences

    def longest_common_subsequence(self, s1, s2):
        """Calculates the length of the longest common subsequence between two strings."""
        if s1 == s2:
            return len(s1)  # Quick return for identical strings
        dp = self._initialize_dp_table(len(s1), len(s2))
        self._calculate_lcs_length(s1, s2, dp)
        return dp[-1][-1]

    def get_lcs_details(self, s1, s2):
        """Retrieves all LCS sequences and determines their contiguity."""
        dp = self._initialize_dp_table(len(s1), len(s2))
        self._calculate_lcs_length(s1, s2, dp)
        memo = {}
        lcs_sequences = self._find_all_lcs_sequences(s1, s2, len(s1), len(s2), dp, memo)
        is_contiguous = any(seq in s1 or seq in s2 for seq in lcs_sequences)
        return lcs_sequences, is_contiguous

# Example usage
if __name__ == "__main__":
    s1, s2 = "it was the best of times", "more money more times"
    ed = EditDistance()
    print("Levenshtein Distance:", ed.levenshtein_distance(s1, s2))
    print("Damerau-Levenshtein Distance:", ed.damerau_levenshtein_distance(s1, s2))
    print("Hamming Distance:", ed.hamming_distance(s1, s2) if len(s1) == len(s2) else "N/A")
    print("Jaro Distance:", ed.jaro_distance(s1, s2))
    print("Jaro-Winkler Distance:", ed.jaro_winkler_distance(s1, s2))
    print("LCS Length:", ed.longest_common_subsequence(s1, s2))
    print("LCS Details:", ed.get_lcs_details(s1, s2))