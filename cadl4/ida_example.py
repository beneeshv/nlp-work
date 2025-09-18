# simple_lda.py
import nltk
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = [
    "Artificial Intelligence is transforming technology.",
    "Climate change poses challenges for global health.",
    "Machine learning improves computer vision applications."
]

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Apply LDA
lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Show topics
print("Identified Topics:")
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")