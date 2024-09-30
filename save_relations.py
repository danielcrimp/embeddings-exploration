import gensim.downloader
from numpy import dot
from numpy.linalg import norm
import numpy as np
import spacy
import h5py

VOCAB_TOPN = 10000

model = gensim.downloader.load('word2vec-google-news-300')

vocab = list(model.index_to_key)

print(f"vocabulary size: {len(vocab)}")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
 
def is_proper_noun(word):
    doc = nlp(word)
    return (doc[0].pos_ == "PROPN" or doc.ents) and doc[0].lemma_ == word

def compute_relationship_vector(word1, word2, model):
    if word1 in model and word2 in model:
        return model[word2] - model[word1]
    return None

filtered_vocab = [word for word in vocab[:VOCAB_TOPN] if is_proper_noun(word)]

print(f"filtered vocabulary size: {len(filtered_vocab)}")

pairs = []
vectors = []

n = len(filtered_vocab)
total_pairs = (n * (n - 1)) // 2  # Maximum number of pairs
print(f"Total possible pairs: {total_pairs}")
pair_counter = 0

for i, word in enumerate(filtered_vocab):
    # for other_word in filtered_vocab[i+1:]:
    for other_word in filtered_vocab:
        relation = compute_relationship_vector(word, other_word, model)
        if relation is not None and word != other_word:
            pair_counter+=1
            pairs.append((word, other_word))
            vectors.append(relation)
            if pair_counter % 1000 == 0:
                print(f"Progress: {pair_counter}/{total_pairs} pairs processed.")


print(f"saved pairs: {len(pairs)}")

vectors = np.array(vectors)

with h5py.File('relationships.h5', 'w') as f:
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset('pairs', data=np.array(pairs, dtype=dt))
    f.create_dataset('vectors', data=vectors)
