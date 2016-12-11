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

reformatted_train_dataset, reformatted_train_labels = reformat(train_dataset, train_labels)
reformatted_valid_dataset, reformatted_valid_labels = reformat(valid_dataset, valid_labels)
reformatted_test_dataset, reformatted_test_labels = reformat(test_dataset, test_labels)
print('reformatted Training set', reformatted_train_dataset.shape, reformatted_train_labels.shape)
print('reformatted Validation set', reformatted_valid_dataset.shape, reformatted_valid_labels.shape)
print('reformatted Test set', reformatted_test_dataset.shape, reformatted_test_labels.shape)

graph = tf.Graph()
set_up_graph_and_run(
    graph,
    reformatted_train_dataset,
    reformatted_train_labels,
    reformatted_valid_dataset,
    reformatted_valid_labels,
    reformatted_test_dataset,
    reformatted_test_labels
)
