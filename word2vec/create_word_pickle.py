from six.moves import cPickle as pickle
from build_dataset import *
from params import *

data, count, dictionary, reverse_dictionary = build_dataset(words)

dataset_pickle_file = 'word.pickle'
try:
  f = open(dataset_pickle_file, 'wb')
  save = {
    'data': data,
    'count': count,
    'dictionary': dictionary,
    'reverse_dictionary': reverse_dictionary,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', dataset_pickle_file, ':', e)
  raise
