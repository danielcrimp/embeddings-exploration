# Embeddings Models as a Knowledge store

Embeddings models allow us to place words as points - vectors - in high-dimensional space - 300 dimensions or even higher. These dimensions represent meaning:

If we were to take a word and translate it in this space, moving it along one (likely many more) of these dimensions, we would change its meaning. In this way, we can take the vector for `'queen'` and adjust it by some vector to return `'king'`. This is a classic example used to explain embeddings spaces, but I feel this notion could be explored further.

This translation in the embeddings space could be likened to drawing a triple, or an edge, in a knowledge graph, or adding a record to a relational database. In the above example, we took two symbols, 'queen' and 'king', and (implicitly) drew a 'gender_counterpart' connection between the two, thus realising a fact.
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

This provides some evidence that at least this model has the semantic precision required to make retrievals of the sort we expect from a traditional database - at least in a very simple example with countries and capitals.

I should note that we've made a bold assumption here that we can determine a universal relation vector for countries and capitals based on a simple average. In reality the relation would vary based on the input vector - in 300 dimensions the odds of the translation being consistent are minute. In short: the similarity search we used to jump to the nearest word is doing a lot of work here.

The incorrect guesses are my favourite - many guesses seem to point toward the most populous city in the country. Anecdotally, I know a lot of humans mistake Auckland and Sydney for capitals. Interestingly, in one case - Peru - the model selects instead the capital of its neighbour, Ecuador.

There are also some cases of simple differences between the spelling of the input data and model vocabulary.

### Experiment 2: Generalised fact retrieval from Embedding Space

In Experiment 1, we showed evidence that facts can be somewhat reliably extracted from embedding space by calculating a relation vector from known relationships and applying it to word vectors from the same semantic category.

In this experiment, I'd like to try and generalise the method in Experiment 1. Rather than primitively bashing together a relation vector by averaging the vector from multiple known class-to-class examples, can we use more traditional data science techniques to bundle facts of similar meaning together?

On paper, we should be able to store a relation vector in the same way we store a word vector. From there, traditional clustering and similarity measures should be applicable.

![finding country:capital facts](./images/country_fact_search.png)

In `save_relations.py`, I've built a script which takes the most popular terms in the model's vocabulary and saves the relation vectors between each, as above. This is a memory-heavy process, so I've used .h5 binaries for storage.

#### Results

`python save_relations.py 20000`

`python explore_relations.py China Beijing`

| left                             | right        |
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

With filtering, of 100 similar examples, we get 24 that are valid pairs. Most of these seem correct.

- Brazil got Rio de Janeiro, rather than Brasilia. Rio de Janeiro is a larger city, and was also the capital until 1960.
- Italy got Turin, rather than Rome. This is unusual, because I don't think the Google News dataset goes back to the 1860s.
- Australia got Sydney, rather than Canberra
- Germany Munich, rather than Berlin
- We got some weird Continent to City pairs returned - Africa and Nairobi, Europe and Athens.

It would seem it's hard to distinguish between Countries/Capitals and other "big land, little land" pairs.

So, a mild success in 'querying' an embeddings model, using an example fact.

#### Experimenting with other fact pairs

Countries and Capitals are an easy pick, especially for a model trained on a News dataset. What about other Relations?

Ideas:
- Sportspeople and their main sports
- Celebrities and their country of origin
- Biological class-to-instance pairs - i.e. canine to dog, feline to cat, murine to mouse
- Gender counterparts

Let's try some.

`python explore_relations.py federer switzerland`

| left        | right   |
|:------------|:--------|
| Switzerland | Roddick |
| Germany     | Agassi  |
| Italy       | Safina  |

...uhhh...

`python explore_relations.py mouse murine`

| left | right    |
|:-----|:---------|
| RA   | keyboard |
| IL   | button   |
| ii   | wizard   |
| RNA  | app      |
| NA   | locate   |

...uh-oh.

Unfortunately, even when trying many different example pairs, we get very poor performance - mostly junk that's returned. My guess is that these relations are much less salient in the embedding space than that of a Country-to-Capital relation. That is, "dog" and "canine" will be mentioned in fewer News articles than countries and capitals, so the vector representing their meaning will be less accurate.

The idea of a signal-to-noise ratio comes to mind here: we get noisier results from less crystallised topics, and clearer results when looking into more culturally prominent entities. 

This Word2Vec model is quite lightweight - a couple of gigabytes and 300 dimensions. It may be that a larger, say, Sentence Transformer type model would capture relationships in more niche topics with more detail (although, we'd then require additional overhead of managing sub-word vocabularies and likely require more computational capability than my laptop).

