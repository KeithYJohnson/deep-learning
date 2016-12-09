from six.moves import cPickle as pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
from data_locations import *
from shuffle_dataset import *

def linear_train(num_examples):
    filename = 'notMNIST.pickle'
    data = pickle.load(open(filename, 'rb'))
    [shuffled_train_data, shuffled_train_labels] = shuffle_dataset(data['train_dataset'], data['train_labels'])
    [shuffled_test_data, shuffled_test_labels]   = shuffle_dataset(data['test_dataset'], data['test_labels'])

    train_data = shuffled_train_data[:num_examples].reshape(num_examples, image_size ** 2)
    test_data  = shuffled_test_data[:num_examples].reshape(num_examples, image_size ** 2)

    train_labels = shuffled_train_labels[:num_examples]
    test_labels  = shuffled_test_labels[:num_examples]

    logit = LogisticRegression().fit(train_data, train_labels)
    print("logit score: ", logit.score(test_data, test_labels))







# Commenting scores from a typical run.
linear_train(50)
# score:  0.52
linear_train(100)
# score:  0.78
linear_train(1000)
# score:  0.832
linear_train(5000)
# logit score:  0.8472
