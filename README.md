# Word2Vec-Model-from-Scratch
We are working on training a word2vec model from scratch, tuning the hyperparameters for word embeddings and then applying the best model to different tasks.

The model training section contains following parts:
1. Generate data batches for skip-gram model in data.py: in the generate batch function, there are following input parameters:
- batch size which is the number of word pairs in each data batch.
- skip windows which decides the range of context words can be taken from.
- num skips determine the total number of context words we will sample for each center word.
In order to generate data batches, we loop through the whole dataset until the batch size is reached. Each center word is located at data index while context words are sampled from index of [data index - skip window, data index + skip window].

2. Implement word-to-vec model in model.py: we train model using following two loss functions:
- cross entropy loss or negative log likelihood loss (NLL). 
- negative sample loss (NEG)

3. Define the training process in train.py: generating data batches and/or negative samples, feeding the data to the word-to-vec model and parameter update via back propogation.
