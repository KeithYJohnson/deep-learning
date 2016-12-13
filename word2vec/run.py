from __future__ import print_function
import math
import numpy as np
import random
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE
from read_data import *
filename = 'text8.zip'
words = read_data(filename)
print('Data size %d' % len(words))
