import numpy as np
import collections
import random
import tensorflow as tf
import math

data_index = 0

def generate_cbow_batch(data, batch_size, cbow_window):
  global data_index
  span = 2 * cbow_window + 1
  batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size):
    buffer_list = list(buffer)
    labels[i, 0] = buffer_list.pop(cbow_window)
    batch[i] = buffer_list
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels
