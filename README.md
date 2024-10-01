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

The incorrect guesses are my favourite - many guesses seem to point toward the most populous city in the country. Anecdotally, I know a lot of humans mistake Auckland and Sydney for capitals. Interestingly, in one case - Peru - the model selects instead the capital of its neighbour, Ecuador.

There are also some cases of simple differences between the spelling of the input data and model vocabulary.

### Experiment 2: Generalised fact retrieval from Embedding Space

In Experiment 1, we showed evidence that facts can be somewhat reliably extracted from embedding space by calculating a relation vector from known relationships and applying it to word vectors from the same semantic category.

In this experiment, I'd like to try and generalise the method in Experiment 1. Rather than primitively (manually) bashing together a relation vector by averaging, can we use more traditional data science techniques to bundle facts of similar meaning together?

On paper, we should be able to store a relation vector in the same way we store a word vector. From there, traditional clustering and similarity measures should be applicable.

/// picture of plotting relation vectors in a new space and clustering

