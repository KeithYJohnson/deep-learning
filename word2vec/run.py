from __future__ import print_function
import math
import numpy as np
import random
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE
from read_data import *
from build_dataset import *
filename = 'text8.zip'
words = read_data(filename)
print('Data size %d' % len(words))

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.
