import gensim.downloader
import h5py
import sys
from sklearn.metrics.pairwise import cosine_similarity

query_left, query_right = sys.argv[1], sys.argv[2]

model = gensim.downloader.load('word2vec-google-news-300')

query_vector = model[query_left] - model[query_right]

with h5py.File('relationships.h5', 'r') as f:
    stored_vectors = f['vectors'][:]
    stored_pairs = f['pairs'][:]

similarities = cosine_similarity([query_vector], stored_vectors)

top_indices = similarities.argsort()[0][-100:][::-1]  # Top 100 most similar relationships
similar_relationships = [stored_pairs[i] for i in top_indices]

decoded_pairs = [(word1.decode('utf-8'), word2.decode('utf-8')) for word1, word2 in stored_pairs[top_indices]]

scanned = {query_left, query_right}

for pair in decoded_pairs:
    if not scanned.intersection(pair):
        scanned.update(pair)
        print(pair)


