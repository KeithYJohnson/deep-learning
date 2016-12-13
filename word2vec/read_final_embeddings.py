from six.moves import cPickle as pickle
from params import *

def read_final_embeddings():
    print('reading embeddings out of pickle file')
    with open(final_embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
        return embeddings
