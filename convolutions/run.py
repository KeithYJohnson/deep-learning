from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
from six.moves import cPickle as pickle
from six.moves import range
sys.path.append('/Users/keithjohnson/courses/deep_learning/fullyconnected')
from params import *
from load_data import *
from reformat_as_cube import *
from run_conv_net import *

[train_dataset,
 train_labels,
 valid_dataset,
 valid_labels,
 test_dataset,
 test_labels] = load_data(pickle_file)

train_dataset, train_labels = reformat_as_cube(train_dataset, train_labels)
valid_dataset, valid_labels = reformat_as_cube(valid_dataset, valid_labels)
test_dataset, test_labels = reformat_as_cube(test_dataset, test_labels)
print('Cube Reformatted Training set', train_dataset.shape, train_labels.shape)
print('Cube Reformatted Validation set', valid_dataset.shape, valid_labels.shape)
print('Cube Reformatted Test set', test_dataset.shape, test_labels.shape)

graph = tf.Graph()

run_conv_net(
    graph,
    train_dataset,
    train_labels,
    valid_dataset,
    valid_labels,
    test_dataset,
    test_labels
)
