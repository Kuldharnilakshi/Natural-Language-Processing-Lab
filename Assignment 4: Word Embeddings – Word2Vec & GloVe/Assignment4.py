# Assignment 4: Word Embeddings - Word2Vec

import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.decomposition import PCA


# Training sentences
sentences = [
    ['natural', 'language', 'processing', 'is', 'a', 'field',
     'of', 'artificial', 'intelligence'],

    ['machine', 'learning', 'algorithms', 'learn',
     'patterns', 'from', 'data'],

    ['deep', 'learning', 'uses', 'neural', 'networks',
     'with', 'multiple', 'layers'],

    ['word', 'embeddings', 'capture', 'semantic',
     'meaning', 'of', 'words'],

    ['transformers', 'use', 'attention', 'mechanism',
     'for', 'sequence', 'modeling'],

    ['bert', 'is', 'a', 'pre', 'trained', 'transformer',
     'model', 'for', 'nlp']
]


# Create Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=100,
    seed=42
)


# Vocabulary size
print("Vocabulary Size:", len(model.wv))


# Similar words
print("\nWords similar to 'neural':")
print(model.wv.most_similar('neural', topn=5))


# Words for visualization
viz_words = [
    'neural',
    'networks',
    'deep',
    'learning',
    'language',
    'nlp'
]

viz_words = [word for word in viz_words if word in model.wv]

vectors = np.array([
    model.wv[word] for word in viz_words
])


# Reduce 100 dimensions to 2 dimensions
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)


# Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    coords[:, 0],
    coords[:, 1],
    s=80
)

for i, word in enumerate(viz_words):
    plt.annotate(
        word,
        (coords[i, 0], coords[i, 1])
    )

plt.title("Word Embeddings - PCA Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)

plt.savefig("embeddings_pca.png")
plt.show()