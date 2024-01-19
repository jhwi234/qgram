import numpy as np

class EditDistance:
    def wagner_fischer(self, s1, s2):
        """Wagner-Fischer (Levenshtein) distance calculation."""
        if s1 == s2: return 0  # Quick return for identical strings
        if len(s1) > len(s2): s1, s2 = s2, s1

        prev_row = np.arange(len(s1) + 1)
        for c2 in s2:
            current_row = [prev_row[0] + 1]
            for i1, c1 in enumerate(s1):
                insertions = prev_row[i1 + 1] + 1
                deletions = current_row[i1] + 1
                substitutions = prev_row[i1] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            prev_row = current_row

        return prev_row[-1]

    def damerau_levenshtein_distance(self, s1, s2):
        """Damerau-Levenshtein distance calculation."""
        if s1 == s2: return 0  # Quick return for identical strings
        d = np.zeros((2, len(s2) + 1), dtype=int)
        d[0, :] = np.arange(len(s2) + 1)
        for i in range(1, len(s1) + 1):
            d[i % 2, 0] = i
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i % 2, j] = min(d[(i - 1) % 2, j] + 1, d[i % 2, j - 1] + 1, d[(i - 1) % 2, j - 1] + cost)
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    d[i % 2, j] = min(d[i % 2, j], d[(i - 2) % 2, j - 2] + cost)
        return d[len(s1) % 2, -1]

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

    def longest_common_subsequence(self, s1, s2):
        """Calculates the length of the longest common subsequence between two strings."""
        if s1 == s2:
            return len(s1)  # Quick return for identical strings
        prev_row = [0] * (len(s2) + 1)
        for c1 in s1:
            current_row = [0]
            for j, c2 in enumerate(s2):
                if c1 == c2:
                    current_row.append(prev_row[j] + 1)
                else:
                    current_row.append(max(current_row[-1], prev_row[j + 1]))
            prev_row = current_row
        return prev_row[-1]
    
    def get_lcs_details(self, s1, s2):
        """Retrieves the LCS sequence and determines its contiguity."""
        # Build the matrix for LCS calculation
        matrix = [["" for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1] + s1[i - 1]
                else:
                    matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)

        # Extract the LCS sequence
        lcs_seq = matrix[-1][-1]

        # Determine contiguity
        is_contiguous = self._check_contiguity(lcs_seq, s1, s2)

        return lcs_seq, is_contiguous

    def _check_contiguity(self, subseq, s1, s2):
        """Determines if a subsequence is contiguous in either of the original strings."""
        return subseq in s1 or subseq in s2


# Example usage
if __name__ == "__main__":
    s1, s2 = "kitten", "sittin"
    ed = EditDistance()
    print("Levenshtein Distance (Wagner-Fischer):", ed.wagner_fischer(s1, s2))
    print("Damerau-Levenshtein Distance:", ed.damerau_levenshtein_distance(s1, s2))
    print("Hamming Distance:", ed.hamming_distance(s1, s2) if len(s1) == len(s2) else "N/A")
    print("Jaro Distance:", ed.jaro_distance(s1, s2))
    print("Jaro-Winkler Distance:", ed.jaro_winkler_distance(s1, s2))
    print("LCS Length:", ed.longest_common_subsequence(s1, s2))
    print("LCS Details:", ed.get_lcs_details(s1, s2))
