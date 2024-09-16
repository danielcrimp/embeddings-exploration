# embeddings-exploration

I had an idea that exploration of vector spaces could be used to repair data quality. That is, if you were to have an embeddings model sufficiently trained on organisational data, you should be able to explore the space to find the model's best guess as to what data might be if it's missing.

For example:

| Point 1     | Point 2     |
| ----------- | ----------- |
| Dog         | Canine      |
| Cat         | Feline      |
| Mouse       | ???         |

We should be able to do **some sort** of traversal in embeddings space - similar to that between Dog and Canine, Cat and Feline, to get from 'Mouse' to Murine.

I'm going to try to do this with a common model, like word2vec, and maybe some others.