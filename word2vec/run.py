from __future__ import print_function
from six.moves import cPickle as pickle
import math
import numpy as np
import random
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE
from read_data import *
from build_dataset import *
from generate_batch import *
from train_skip_gram import *
from params import *
from read_final_embeddings import *
from plot import *
from generate_cbow_batch import *
from train_cbow import *

with open(dataset_pickle_file, 'rb') as f:
    save = pickle.load(f)
    data = save['data']
    count = save['count']
    dictionary = save['dictionary']
    reverse_dictionary = save['reverse_dictionary']
    del save  # hint to help gc free up memory
    print('len(data): ', len(data))
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])

# for num_skips, skip_window in [(2, 1), (4, 2)]:
#     data_index = 0
#     batch, labels = generate_batch(data, batch_size=8, num_skips=num_skips, skip_window=skip_window)
#     print('len batch: ', len(batch))
#     print('len labels: ', len(labels))
#     print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
#     print('    batch:', [reverse_dictionary[bi] for bi in batch])
#     print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

for cbow_window in [1, 2]:
    batch, labels = generate_cbow_batch(data, 12, cbow_window)
    print(batch)
    print(labels)

# final_embeddings = train_skip_gram(data,reverse_dictionary)
# f = open(final_embeddings_file, 'wb')
# pickle.dump(final_embeddings, f, pickle.HIGHEST_PROTOCOL)

cbow_final_embeddings = train_cbow(data,reverse_dictionary)
f = open(cbow_final_embeddings_file, 'wb')
pickle.dump(cbow_final_embeddings, f, pickle.HIGHEST_PROTOCOL)
#
num_points = 400

print('final_embeddings: ', cbow_final_embeddings)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(cbow_final_embeddings[1:num_points+1, :])

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
