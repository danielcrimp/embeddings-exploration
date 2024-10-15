# Embeddings Models as a Knowledge store

Embeddings models allow us to place words as points - vectors - in high-dimensional space - 300 dimensions or even higher. These dimensions represent meaning:

If we were to take a word and translate it in this space, moving it along one (likely many more) of these dimensions, we would change it's meaning. In this way, we can take the vector for `'queen'` and adjust it by some vector to return `'king'`. This is a classic example used to explain embeddings spaces, but I feel this notion could be explored further.

This translation in the embeddings space could be likened to drawing a triple, or an edge, in a knowledge graph, or adding a record to a relational database. We take two nodes, 'queen' and 'king', and (implicitly) draw a 'gender_counterpart' connection between the two, thus realising a fact.
Could we extend this analogy and somehow interact with embeddings spaces as a traditional knowledge store?
How would we query it? i.e.:

`SELECT female_term, male_term FROM gendered_words`

What would the advantages of this arrangement be?
How could such a model be trained for arbitrary use-cases, like enterprise?

### Experiment 1: Proving the basic idea

ML models are finicky and often don't behave how we expect. Our mind pictures the happy path, and clean results magically appearing. To mitigate this, I'm going to start with the simplest example.

I elected to choose a pretrained word2vec model, `'word2vec-google-news-300'`. As the name suggests, it is trained on a dataset scraped from Google News.

In `countries_to_capitals.py`, I've built a script which takes twenty country-to-capital pairs and averages the vectors relating each, attempting to yield a vector which could be used to retrieve the Capital city of a wider sample of countries.

![relating countries to capitals](./images/country_to_capital.png)

#### Results

<p style="font-size: 12px;">

| Country                          | Capital       | guess_capital  |
|----------------------------------|---------------|----------------|
| China                            | Beijing       | Beijing        |
| India                            | New_Delhi     | Delhi          |
| United_States                    | WashingtonDC  | Washington_DC  |
| Indonesia                        | Jakarta       | Jakarta        |
| Pakistan                         | Islamabad     | Islamabad      |
| **... (40 records omitted) ...** |
| Yemen                            | Sana'a        | Sana'a         |
| Madagascar                       | Antananarivo  | Antananarivo   |
| North_Korea                      | Pyongyang     | Pyongyang      |
| Australia                        | Canberra      | Sydney         |
| Ivory_Coast                      | Yamoussoukro  | Abidjan        |

**Number of correct guesses:** 39 out of 51 records.

</p>

#### Thoughts

This is a dinky example, but provides decent evidence that at least this model has the semantic resolution(?) required to make retrievals of the sort we expect from a traditional database.

I should note that we've made a bold assumption here that we can determine a universal relation vector for countries and capitals based on a simple average. In reality the relation would **have** to vary based on the input vector. In short - the similarity search we used to jump to the nearest word is doing a lot of work here.

The incorrect guesses are my favourite - many guesses seem to point toward the most populous city in the country. Anecdotally, I know a lot of humans mistake Auckland and Sydney for capitals. Interestingly, in one case - Peru - the model selects instead the capital of its neighbour, Ecuador.

There are also some cases of simple differences between the spelling of the input data and model vocabulary.

### Experiment 2: Generalised fact retrieval from Embedding Space

In Experiment 1, we showed evidence that facts can be somewhat reliably extracted from embedding space by calculating a relation vector from known relationships and applying it to word vectors from the same semantic category.

In this experiment, I'd like to try and generalise the method in Experiment 1. Rather than primitively (manually) bashing together a relation vector by averaging, can we use more traditional data science techniques to bundle facts of similar meaning together?

On paper, we should be able to store a relation vector in the same way we store a word vector. From there, traditional clustering and similarity measures should be applicable.

![finding country:capital facts](./images/country_fact_search.png)


#### Results

| Country                          | Capital      |
|:---------------------------------|:-------------|
| Russia                           | Moscow       |
| Korea                            | Seoul        |
| Malaysia                         | Kuala_Lumpur |
| Japan                            | Tokyo        |
| Turkey                           | Ankara       |
| **... (14 records omitted) ...** |              |
| India                            | Delhi        |
| Thailand                         | Bangkok      |
| Egypt                            | Cairo        |
| Pakistan                         | Islamabad    |
| Indonesia                        | Jakarta      |

#### Thoughts

With filtering, of 100 similar examples, we get 24 that are valid pairs. Most of these seem correct. This is 

- Brazil got Rio de Janeiro, rather than Brasilia. Rio de Janeiro is a larger city, and was also the capital until 1960.
- Italy got Turin, rather than Rome. This is unusual, because I don't think the Google News dataset goes back to the 1860s.
- Australia got Sydney, rather than Canberra
- Germany Munich, rather than Berlin
- We got some weird Continent to City pairs returned - Africa and Nairobi, Europe and Athens.

It would seem it's hard to distinguish between Countries/Capitals and other "big land, little land" pairs.

#### Experimenting with other fact pairs

Countries and Capitals are an easy pick, especially for a model trained on a News dataset. What about other Relations?

Ideas:
- Sportspeople and their main sports
- Celebrities and their country of origin
- Biological class-to-instance pairs - i.e. canine to dog, feline to cat, murine to mouse
- Gender counterparts

Unfortunately, even when trying many different example pairs, we get very poor performance - mostly junk that's returned. I draw this down to that these relations are much less salient in the embedding space than that of a Country-to-Capital relation. Intuition tells me that the words "dog" and "canine" will be covered less in News articles, so their semantic meaning will be less crystallised than the likes of Countries and Capitals.

This Word2Vec model is also quite lightweight - a couple of gigabytes and 300 dimensions. To accurately express the meaning of 3M symbols with such a small representation would be impressive - indeed I find it impressive that it performs as well as it does. It may be that a larger, say, Sentence Transformer type model would capture relationships in more niche topics with more detail (although, we'd then require additional overhead of managing sub-word vocabularies and likely require more computational capability than my laptop).

The idea of a signal-to-noise ratio comes to mind here: we get noisier results from less crystallised topics in w2v, and clearer results when looking into culturally prominent entities. 
