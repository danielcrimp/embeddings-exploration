import gensim.downloader
from numpy import dot
from numpy.linalg import norm
import numpy as np
import spacy

model = gensim.downloader.load('word2vec-google-news-300')

print(f"vocabulary size: {len(model.index_to_key)}")

relation_vector = {}

similarity_ceiling = 0.95
similarity_floor = 0.80
edge_similarity_threshold = 0.75
vocab_start = 300
vocab_end = 1000
vocab_topn = 150000

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def is_noun_and_singular_and_lowercase(word):
    doc = nlp(word)
    token = doc[0]
    return token.pos_ == "NOUN" and token.lemma_ == word and word.lower() == word


words_to_process = [word for word in model.index_to_key[:vocab_topn] if is_noun_and_singular_and_lowercase(word)]

print(f"words selected for processing: {len(words_to_process)}")

relations = []
for i, word in enumerate(words_to_process):
    word_vec = model[word]

    most_similar = model.most_similar(positive=[word], topn=100)

    for similar_word, score in most_similar:
        in_range = (similarity_floor <= score <= similarity_ceiling)
        in_words = similar_word in words_to_process
        not_same = similar_word != word

        if in_range and in_words and not_same:
            relations.append((word, similar_word, word_vec - model[similar_word]))
            print([word, similar_word])

print(f"relationships identified: {len(relations)}")

# save to npz file for further exploration
string_pairs = np.array([(x[0], x[1]) for x in relations], dtype='U')
vectors = np.array([x[2] for x in relations])
np.savez('relation_vectors1.npz', string_pairs=string_pairs, vectors=vectors)
