import nltk
from nltk.corpus import inaugural
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download NLTK data (if not already downloaded)
nltk.download('inaugural')

# Load the inaugural speech corpus
corpus = inaugural.raw()

# Tokenize the text
words = word_tokenize(corpus)

# Calculate word frequencies
fdist = FreqDist(words)

# Print the 10 most common words and their frequencies
print("Top 10 Most Common Words:")
common_words = fdist.most_common(10)
for word, freq in common_words:
    print(f"{word}: {freq}")

# Plot a frequency distribution of the 20 most common words
fdist.plot(20, cumulative=False)
