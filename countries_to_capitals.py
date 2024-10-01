import gensim.downloader
import pandas as pd
import numpy as np 


# load csv of countries and their capitals
countries = pd.read_csv('countries.csv')

model = gensim.downloader.load('word2vec-google-news-300')

vector_dim = model.vector_size  
accumulated_vector = np.zeros(vector_dim)

# calculate the average vector relating Countries to their Capitals
vec_count = 0 
for idx, row in countries.head(20).iterrows():
    if row['Country'] in model and row['Capital'] in model:
        vec_count += 1
        accumulated_vector += (model[row['Capital']] - model[row['Country']])

relation_vector = accumulated_vector / vec_count


def check_country(country: str, model, autocorrect_vector):
    if country in model:
        similar_words = model.similar_by_vector(model[country] + autocorrect_vector, topn=10)
        for word, similarity in similar_words:
            if word != country:
                return word
    return None


countries['guess_capital'] = countries['Country'].apply(lambda x: check_country(x, model, relation_vector))
matches = countries[countries['Capital'] == countries['guess_capital']].shape[0]

print(countries)
print(f"\nNumber of correct guesses: {matches} out of {countries.shape[0]} records.")
