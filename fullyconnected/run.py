from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from params import *
from load_data import *
from reformat import *
from set_up_graph_and_run import *

[train_dataset,
 train_labels,
 valid_dataset,
 valid_labels,
 test_dataset,
 test_labels] = load_data(pickle_file)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

graph = tf.Graph()
set_up_graph_and_run(graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
