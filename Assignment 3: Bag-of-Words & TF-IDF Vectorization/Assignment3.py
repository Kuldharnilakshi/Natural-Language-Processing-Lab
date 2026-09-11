import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Corpus
corpus = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks for pattern recognition",
    "Natural language processing handles text and speech data",
    "NLP techniques include tokenization and named entity recognition",
    "Artificial intelligence is transforming healthcare and finance",
    "Neural networks are inspired by the human brain structure"
]

labels = ["ML", "DL", "NLP", "NLP", "AI", "DL"]

# ---------------- Bag of Words (Scratch) ----------------
all_words = [word for doc in corpus for word in doc.lower().split()]
vocabulary = sorted(set(all_words))

bow_matrix = [
    [Counter(doc.lower().split()).get(word, 0) for word in vocabulary]
    for doc in corpus
]

print("BoW Matrix Shape:", len(bow_matrix), "x", len(vocabulary))

# ---------------- TF-IDF (Scratch) ----------------
N = len(corpus)

tf_matrix = [
    [Counter(doc.lower().split()).get(word, 0) / len(doc.split()) for word in vocabulary]
    for doc in corpus
]

idf_vector = [
    math.log(N / sum(1 for doc in corpus if word in doc.lower().split())) + 1
    for word in vocabulary
]

tfidf_scratch = np.array(tf_matrix) * np.array(idf_vector)

# ---------------- Sklearn Verification ----------------
cv = CountVectorizer()
bow_sklearn = cv.fit_transform(corpus).toarray()

tv = TfidfVectorizer()
tfidf_sklearn = tv.fit_transform(corpus).toarray()

print("Sklearn BoW Shape:", bow_sklearn.shape)
print("Sklearn TF-IDF Shape:", tfidf_sklearn.shape)

# ---------------- Top 5 Keywords ----------------
feature_names = tv.get_feature_names_out()

print("\nTop 5 Keywords by Class\n")

for label in sorted(set(labels)):
    index = [i for i, l in enumerate(labels) if l == label]
    mean_tfidf = tfidf_sklearn[index].mean(axis=0)
    top = mean_tfidf.argsort()[-5:][::-1]

    print(label, ":", [feature_names[i] for i in top])