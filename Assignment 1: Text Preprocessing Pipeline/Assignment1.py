import nltk
import spacy
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input text
text = """
Natural language processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence.
NLP techniques are widely used in sentiment analysis,
machine translation, chatbots, and information retrieval.
"""

# -------------------- NLTK --------------------

# Tokenization
tokens = word_tokenize(text)

# Stop-word removal
stop_words = set(stopwords.words("english"))
filtered_nltk = [
    word for word in tokens
    if word.lower() not in stop_words and word.isalpha()
]

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_nltk]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_nltk]

# POS Tagging
pos_nltk = nltk.pos_tag(filtered_nltk)

# -------------------- spaCy --------------------

doc = nlp(text)

filtered_spacy = [
    token.text for token in doc
    if not token.is_stop and token.is_alpha
]

lemmas_spacy = [
    token.lemma_ for token in doc
    if not token.is_stop and token.is_alpha
]

pos_spacy = [
    (token.text, token.pos_)
    for token in doc
    if not token.is_stop and token.is_alpha
]

# -------------------- Comparison Table --------------------

comparison = pd.DataFrame({
    "Word": filtered_nltk[:10],
    "NLTK Stem": stemmed[:10],
    "NLTK Lemma": lemmatized[:10],
    "spaCy Lemma": lemmas_spacy[:10],
    "NLTK POS": [tag[1] for tag in pos_nltk[:10]],
    "spaCy POS": [tag[1] for tag in pos_spacy[:10]]
})

print("\nTEXT PREPROCESSING COMPARISON\n")
print(comparison.to_string(index=False))