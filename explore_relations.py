import gensim.downloader
import h5py
from sklearn.metrics.pairwise import cosine_similarity

query_left = 'Beijing'
query_right = 'China'

model = gensim.downloader.load('word2vec-google-news-300')

query_vector = model[query_left] - model[query_right]

with h5py.File('relationships.h5', 'r') as f:
    stored_vectors = f['vectors'][:]
    stored_pairs = f['pairs'][:]

similarities = cosine_similarity([query_vector], stored_vectors)

top_indices = similarities.argsort()[0][-100:][::-1]  # Top 10 most similar relationships
similar_relationships = [stored_pairs[i] for i in top_indices]

pairs = []

for pair in similar_relationships:
    word1, word2 = pair
    word1 = word1.decode('utf-8')
    word2 = word2.decode('utf-8')
    pairs.append((word1, word2))


scanned = set([query_left, query_right])

for pair in pairs:
    if pair[0] not in scanned and pair[1] not in scanned:
        scanned.update(pair)
        print(pair)


