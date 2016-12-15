import numpy as np
import random

vocabulary_size = 50000
dataset_pickle_file = 'word.pickle'
final_embeddings_file = 'final_embeddings_file.pickle'
final_embeddings_file = 'cbow_final_embeddings_file.pickle'

## Training params
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.
num_steps = 100001
cbow_window = 1
