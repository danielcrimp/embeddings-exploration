import gensim.downloader
from numpy import dot
from numpy.linalg import norm
import spacy

model = gensim.downloader.load('word2vec-google-news-300')

print(len(model.index_to_key))

relation_vector = {}

similarity_ceiling = 0.85
similarity_floor = 0.5
edge_similarity_threshold = 0.75
vocab_start = 300
vocab_end =  1000
vocab_topn = 1000

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

relations = []

# Load spaCy's small English model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Function to check if a word is a noun
def is_noun(word):
    doc = nlp(word)
    # Check the first token in the word and its POS tag
    return doc[0].pos_ == "NOUN"

nouns = [x for x in model.index_to_key if is_noun(x)]

print(len(nouns))

for word in model.index_to_key[vocab_start:vocab_end]:
    for other_word in model.index_to_key[:vocab_topn]:
        word_vec = model[word]
        other_word_vec = model[other_word]
        both_nouns = is_noun(word) and is_noun(other_word)
        if (word != other_word) and (similarity_floor <= cosine_similarity(word_vec, other_word_vec) <= similarity_ceiling) and both_nouns:
            relations.append((word, other_word, word_vec-other_word_vec))


# print(relations)  

relation_relations = []

for edge in relations:
    for other_edge in relations:
        edge_vec = edge[2]
        other_edge_vec = other_edge[2]
        woids = [edge[0],edge[1], other_edge[0], other_edge[1]]
        if (edge != other_edge) and (cosine_similarity(edge_vec, other_edge_vec) > edge_similarity_threshold) and len(set(woids)) == len(woids):
            relation_relations.append((edge[0],edge[1], other_edge[0], other_edge[1]))

print(relation_relations)
