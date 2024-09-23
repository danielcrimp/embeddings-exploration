import gensim.downloader
from numpy import dot
from numpy.linalg import norm

model = gensim.downloader.load('word2vec-google-news-300')

print(len(model.index_to_key))

relation_vector = {}

similarity_threshold = 0.5
edge_similarity_threshold = 0.75
vocab_start = 3000
vocab_end =  4000
vocab_topn = 4000

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

relations = []

for word in model.index_to_key[vocab_start:vocab_end]:
    for other_word in model.index_to_key[:vocab_topn]:
        word_vec = model[word]
        other_word_vec = model[other_word]
        if (word != other_word) and (cosine_similarity(word_vec, other_word_vec) > similarity_threshold):
            relations.append((word, other_word, word_vec-other_word_vec))


# print(relations)  

relation_relations = []

for edge in relations:
    for other_edge in relations:
        edge_vec = edge[2]
        other_edge_vec = other_edge[2]
        if (edge != other_edge) and (cosine_similarity(edge_vec, other_edge_vec) > edge_similarity_threshold):
            relation_relations.append((edge[0],edge[1], other_edge[0], other_edge[1]))

print(relation_relations)
