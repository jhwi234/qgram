### Linguistic Laws

#### Zipf's Law
**Core Idea**: The frequency of a word is inversely proportional to its rank in a frequency table.

**Formula**:
$$ f(r) = \frac{C}{r^\alpha} $$
- `f(r)`: Frequency of the word at rank `r`.
- `C`: Constant, approximating the frequency of the most common word.
- `alpha`: Exponent, empirically tends to approximately 1.

**Implication**: Higher-ranked words are more frequent; as rank number increases, token frequency decreases.

#### Zipf-Mandelbrot Law
**Core Idea**: A generalization of Zipf's Law that introduces a shift parameter `q` and a scaling parameter `s`.

**Formula**:
$$ f(k; N, q, s) = \frac{1}{(k + q)^s \cdot H_{N,q,s}} $$
- `k`: Rank of an element e.g., a word type.
- `N`: Total number of elements e.g., word tokens
- `q`: Shift parameter, adjusting the starting point of the frequency distribution.
- `s`: Scaling parameter, modifying the slope of the frequency distribution.
- `H_{N,q,s}`: Normalizing constant (generalized harmonic number).

**Key Properties**:
- **`q` (Shift Parameter)**: Adjusts the starting point of the frequency distribution, accounting for the initial offset in the ranking of word frequencies. This parameter helps in modeling the distribution of the most common words more accurately.
- **`s` (Scaling Parameter)**: Modifies the slope of the frequency distribution, reflecting the rate at which word frequencies decrease with rank. A higher `s` value indicates a steeper decline, signifying a greater disparity in word usage.
- The inclusion of `H_{N,q,s}`, a normalizing constant, ensures that the distribution is properly scaled across different corpus sizes and compositions.

#### Heaps' Law
**Core Idea**: Vocabulary size grows with corpus size, but at a decreasing rate.

**Formula**:
$$ V(R) = K \cdot R^\beta $$
- `V(R)`: Vocabulary size when corpus size is `R`.
- `K`: Constant representing initial rate of vocabulary growth.
- `beta`: Rate at which new words appear as corpus size increases.

**Interpretation**: As the corpus size increases, the rate at which new words are added to the vocabulary diminishes at a rate of `beta`.

**Application**: Useful for estimating author vocabulary and differentiating between human and Transformer Model texts (Lai, Randhawa, & Sheridan, 2023).

[Heaps' Law in GPT-Neo Large Language Model Emulated Corpora](obsidian://open?vault=Obsidian%20Vault&file=PDFs%2FHeaps_Law_in_GPT-Neo_Large_Language_Model_Emulated_Corp.pdf)

#### Yule's K
**Core Idea**: Measures lexical diversity in a text corpus.

**Formula**:
$$ K = 10^4 \left( \frac{\sum_{i=1}^{V} f_i^2 - N}{N^2} \right) $$
- `f_i`: (Frequency of the `i^{th}` word) Represents how many times each unique word appears in the corpus, which is crucial for understanding the distribution of word usage.
- `V`: The count of distinct words in the corpus, the vocabulary size.
- `N`: The overall word count, including repetitions, denotes the corpus size.

**Interpretation**: Yule's K Measure quantifies lexical diversity by calculating the variation in word frequency distribution. A higher `K` value suggests a text with a wide range of unique words, implying rich lexical diversity. Conversely, a lower `K` indicates more repetitive use of a smaller set of words. This measure is particularly useful for comparative studies, such as contrasting authorial styles or the complexity of texts, rather than providing an absolute standard of diversity.








### Zipf's Law and Related Concepts

#### Zipf's Law Overview
**Definition**: An empirical law where the nth entry's value in a decreasingly sorted list of values is inversely proportional to n.

**Application in Linguistics**: Applies to word frequencies in natural language texts. The most common word occurs roughly twice as often as the second, three times as often as the third, and so on.

**Example**: In the Brown Corpus, "the" appears most frequently, making up about 7% of all word occurrences.

#### Zipf-Mandelbrot Law
**Modified Form of Zipf's Law**: Incorporates additional parameters `a` and `b` for better fit, with `a` approximately 1 and `b` approximately 2.7.

**Formula**:
$$ \text{frequency} \propto \frac{1}{(\text{rank} + b)^a} $$

#### Mathematical Formalization
**Probability Mass Function (PMF)**: For an element of rank k in a distribution of N elements,
$$ f(k;N) = \frac{1}{H_{N}}\frac{1}{k} $$
where `H_{N}` is the Nth harmonic number.

**Generalized Version**: Uses an inverse power law with exponent `s`,
$$ f(k;N,s) = \frac{1}{H_{N,s}}\frac{1}{k^s} $$
where `H_{N,s}` is the generalized harmonic number.

**Extension to Infinity**: Valid for `s > 1`, where `H_{N,s}` becomes the Riemann' zeta function.

#### Generalized Harmonic Number
**Role in Zipf's Law**: Normalizes frequencies so their sum equals the total number of words in a corpus.

**Formula**:
$$ H(N, \alpha) = \sum_{i=1}^{N} \frac{1}{i^\alpha} $$
- `N`: Number of terms (vocabulary size).
- `alpha`: Exponent in Zipf's Law.

#### Empirical Testing
**Method**: Using a Kolmogorov–Smirnov test to fit empirical distribution to the hypothesized power law.

**Visualization**: Plotted on a log-log graph with rank order and frequency logarithms. Conformity to Zipf's law indicated by linear function with slope `-s`.

#### Statistical Explanations
- **Random Text Analysis**: Shows that randomly generated words can follow Zipf's law's macro-trend.
- **Taylor Series Truncation**: First-order truncation results in Zipf's law, second-order in Mandelbrot's law.
- **Principle of Least Effort**: Proposes that linguistic efficiency leads to Zipf distribution.
- **Random Typing Model**: Suggests that random typing by monkeys can produce words following Zipf's law.
- **Preferential Attachment Process**: "Rich get richer" dynamic can result in Yule–Simon distribution, fitting word frequency and city rank better than Zipf's law.
- **Atlas Models**: Systems of exchangeable diffusion processes can mathematically yield Zipf's law.

#### Related Laws
**Zipf–Mandelbrot Law**: A generalized form with frequencies 
$$ f(k;N,q,s) = \frac{1}{C}\frac{1}{(k+q)^s} $$
**Relation to Pareto Distribution**: Zipfian distributions derived from Pareto distributions by variable exchange.

**Yule–Simon Distribution Tail Frequencies**: Approximate 
$$ f(k;\rho) \approx \frac{[constant]}{k^{\rho +1}} $$
**Parabolic Fractal Distribution**: Improves fit over simple power-law relationships.

**Connection with Benford's Law**: Benford's law seen as a bounded case of Zipf's law.

#### Applications
- **Word Frequency in Text**: Power-law distribution of word frequencies in text corpora.

**Harmonic Series and Zipf–Mandelbrot Generalization**: Addresses the fixed vocabulary size issue, showing different parameter behaviors for functional and contentive words.

The equation that incorporates vocabulary size, corpus token size, and the parameters C, alpha, s, and q in the context of the Zipf

-Mandelbrot law can be formulated as follows:

Let's denote:
- `V` as the vocabulary size,
- `T` as the total corpus token size,
- `C` as a constant,
- `alpha` as the Zipf-Mandelbrot exponent (similar to s in the original formulation),
- `q` as the shift parameter in the Zipf-Mandelbrot distribution.

The frequency `f(k)` of a word at rank `k` in the Zipf-Mandelbrot distribution can be given by:
$$ f(k) = \frac{C}{(k + q)^\alpha} $$

However, to integrate this with the vocabulary size and corpus token size, we need to consider the normalization of the distribution across all words in the vocabulary. This can be achieved by ensuring the sum of frequencies of all words equals the total corpus token size `T`. Therefore, we need to find a constant `C` that satisfies this condition:
$$ T = \sum_{k=1}^{V} \frac{C}{(k + q)^\alpha} $$

This equation represents the balance between the frequency distribution of words (according to Zipf-Mandelbrot law) and the total size of the corpus, taking into account the vocabulary size and the specific parameters of the distribution. Solving this equation for `C` in terms of `T`, `V`, `alpha`, and `q` can be complex and might require numerical methods.

To solve the equation for the constant `C` in the context of the Zipf-Mandelbrot law, given the corpus token size `T` and the vocabulary size `V`, along with the parameters `alpha` and `q`, we need to approach it numerically. The equation is as follows:
$$ T = \sum_{k=1}^{V} \frac{C}{(k + q)^\alpha} $$

This equation can be challenging to solve analytically due to the summation and the nature of the variables involved. Therefore, we use numerical methods to find the value of `C` that satisfies this equation. One common approach is to use a numerical optimization technique, such as a binary search or a root-finding algorithm.






### Linguistic Laws

#### Steps for Solving Using a Root-Finding Algorithm
1. **Define the Function**: 
   - The function `f(C)` is defined as the difference between the sum of frequencies of all words and the total corpus token size `T`:
     $$ f(C) = \left( \sum_{k=1}^{V} \frac{C}{(k + q)^\alpha} \right) - T $$
   - The goal is to find the value of `C` for which `f(C) = 0`.

2. **Choose a Root-Finding Algorithm**: 
   - Employ common algorithms like the Newton-Raphson method or the bisection method, which iteratively adjust the value of `C` to find the root of `f(C)`.

3. **Iterate to Find the Root**: 
   - Begin with an initial guess for `C`, and iteratively update this guess using the chosen algorithm based on the function value and its derivatives, if necessary.

4. **Check for Convergence**: 
   - Continue the process until the function value `f(C)` is sufficiently close to zero, which indicates that the value of `C` balances the equation.

5. **Output `C`**: 
   - When the algorithm converges, the current value of `C` is the solution to the equation.

#### Parameters in Zipf-Mandelbrot Law
- **The Exponent `alpha`**:
  - **Word Frequency Distribution**: Reflects the steepness of the word frequency distribution curve. A higher `alpha` value suggests a few words are extremely common, while a lower `alpha` value indicates a more uniform distribution of word frequencies.
  - **Text Complexity and Style**: A text with a higher `alpha` might be less complex or have a more limited vocabulary, as seen in specialized or technical texts. In contrast, a lower `alpha` may suggest a text has richer vocabulary and more complex sentence structures, typically found in literary works.

- **The Shift Parameter `q`**:
  - **Adjustment for High-Frequency Words**: Helps to more accurately model the behavior of the most frequent words, especially if they do not follow the expected mathematical relationship strictly.
  - **Indicative of Language or Genre**: The value of `q` can characterize different languages or genres, reflecting their unique structural features.

- **The Constant `C`**:
  - **Normalization Factor**: Ensures that the predicted frequencies from the model sum up to the actual size of the corpus, adjusted based on the corpus' total word count and vocabulary size.
  - **Comparative Analysis**: By comparing the value of `C` across different corpora, one can draw conclusions about how word frequencies are distributed relative to the size of the corpus.