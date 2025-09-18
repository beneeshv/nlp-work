# nltk_test.py
# =====================================
# Text Preprocessing using NLTK
# =====================================

import nltk  # ← CHANGE THIS LINE from 'nltk_test' to 'nltk'
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download resources (only needed once)
nltk.download('punkt')  # ← CHANGE THIS from 'nltk_test' to 'nltk'
nltk.download('stopwords')  # ← CHANGE THIS
nltk.download('wordnet')  # ← CHANGE THIS

# Sample Text Corpus
corpus = [
    "The new AI model is revolutionizing natural language processing.",
    "Students are studying hard for their upcoming exams.",
    "The football match was exciting and thrilling to watch!"
]

print("\n🔹 Original Corpus:")
for i, doc in enumerate(corpus, 1):
    print(f"{i}. {doc}")

# 1. Tokenization
print("\n🔹 Tokenization:")
for i, doc in enumerate(corpus, 1):
    tokens = word_tokenize(doc)
    print(f"{i}. {tokens}")

# 2. Stopword Removal
stop_words = set(stopwords.words('english'))
print("\n🔹 Stopword Removal:")
for i, doc in enumerate(corpus, 1):
    tokens = word_tokenize(doc)
    filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
    print(f"{i}. {filtered}")

# 3. Stemming
stemmer = PorterStemmer()
print("\n🔹 Stemming:")
for i, doc in enumerate(corpus, 1):
    tokens = word_tokenize(doc)
    stems = [stemmer.stem(w) for w in tokens if w.isalpha()]
    print(f"{i}. {stems}")

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
print("\n🔹 Lemmatization:")
for i, doc in enumerate(corpus, 1):
    tokens = word_tokenize(doc)
    lemmas = [lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalpha()]
    print(f"{i}. {lemmas}")