from six.moves import cPickle as pickle
import numpy as np

def count_overlaps():
    print("looping through test and train datasets to detect overlap")
    filename = 'notMNIST.pickle'
    data = pickle.load(open(filename, 'rb'))
    train_data = data['train_dataset']
    test_data = data['test_dataset']

    num_copies = 0
    copy_pairs = []
    for train_img in train_data:
        for test_img in test_data:
            if np.array_equal(train_img, test_img):
                print("found an overlap!")
                num_copies += 1
                copy_pairs.append((train_img, test_img))

    print("Found {} instances of overlap".format(num_copies))


count_overlaps()
